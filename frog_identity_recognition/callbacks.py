############################################################
#  Custom Callbacks
############################################################
from keras.callbacks import Callback
from mrcnn.model import load_image_gt, mold_image, MaskRCNN
from mrcnn.utils import compute_ap, Dataset
import numpy as np


class MeanAveragePrecisionCallback(Callback):
    def __init__(self, train_model: MaskRCNN, inference_model: MaskRCNN, dataset: Dataset,
                 calculate_map_at_every_X_epoch=5, dataset_limit=None,
                 verbose=1):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):

        if epoch > 2 and (epoch + 1) % self.calculate_map_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._load_weights_for_model()

            mAPs = self._calculate_mean_average_precision()
            mAP = np.mean(mAPs)

            if logs is not None:
                logs["val_mean_average_precision"] = mAP

            self._verbose_print("mAP at epoch {0} is: {1}".format(epoch + 1, mAP))

        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path,
                                          by_name=True)

    def _calculate_mean_average_precision(self):
        mAPs = []

        # Use a random subset of the data when a limit is defined
        np.random.shuffle(self.dataset_image_ids)

        for image_id in self.dataset_image_ids[:self.dataset_limit]:
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(self.dataset, self.inference_model.config,
                                                                             image_id)
            molded_images = np.expand_dims(mold_image(image, self.inference_model.config), 0)
            results = self.inference_model.detect(molded_images, verbose=0)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                     r["class_ids"], r["scores"], r['masks'])
            mAPs.append(AP)

        return np.array(mAPs)
