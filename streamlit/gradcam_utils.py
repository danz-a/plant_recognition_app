"""Live Grad-CAM for the demo page.

Works directly on the existing layer objects of the loaded model
(backbone -> GAP -> dense) - no graph rebuild, which is exactly the
Keras-3 pitfall that produced wrong heatmaps in early notebook runs.

Every call verifies itself: the probabilities computed on the Grad-CAM
path must match the model's own prediction. If they do not - or if
anything else goes wrong - the caller gets None and shows a note; the
prediction itself is never affected.
"""
import numpy as np

from model_utils import IMG_SIZE, load_bundle


def gradcam_overlay(pil_image, alpha=0.45):
    """PIL image -> (overlay PIL image, class_name, probability) or None.

    Heat = evidence for the *predicted* class, computed at the DenseNet
    final 4x4 activation grid and upsampled - the same recipe as the
    pre-computed panels in section 5.
    """
    try:
        import tensorflow as tf

        model, class_names, _ = load_bundle()
        backbone = model.get_layer("densenet121")
        gap      = model.get_layer("global_average_pooling2d")
        dense    = model.get_layer("dense")

        img  = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x    = tf.constant(np.asarray(img, dtype="float32")[None, ...])

        # The exported graph applies densenet preprocess_input (/255, -mean,
        # /std) as op-layers between augment and the backbone. Calling the
        # backbone directly skips them, so we apply the identical transform
        # here - and the equivalence guard below proves it per call.
        x_pre = tf.keras.applications.densenet.preprocess_input(tf.identity(x))

        with tf.GradientTape() as tape:
            feats = backbone(x_pre, training=False)  # (1, 4, 4, 1024)
            tape.watch(feats)
            preds = dense(gap(feats))                # dropout inactive at inference
            idx   = int(tf.argmax(preds[0]))
            score = preds[:, idx]

        # equivalence guard: this path must agree with the model itself
        ref = model.predict(x.numpy(), verbose=0)
        if not np.allclose(preds.numpy(), ref, atol=1e-4):
            return None
        if int(ref[0].argmax()) != idx:
            return None

        grads = tape.gradient(score, feats)          # (1, 4, 4, 1024)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))          # (1024,)
        cam     = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
        cam     = cam.numpy()
        if cam.max() <= 0:
            return None
        cam /= cam.max()

        # upsample 4x4 -> IMG_SIZE and colour it (jet), blend over the photo
        from PIL import Image as PILImage
        from matplotlib import cm

        heat = PILImage.fromarray(np.uint8(cam * 255)).resize(
            (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR
        )
        heat_rgb = np.uint8(cm.jet(np.asarray(heat) / 255.0)[..., :3] * 255)
        base     = np.asarray(img, dtype="float32")
        overlay  = np.uint8((1 - alpha) * base + alpha * heat_rgb)

        return PILImage.fromarray(overlay), class_names[idx], float(ref[0][idx])
    except Exception:
        return None
