# glaciers-transfer-learning

The **calving front** of a glacier signifies whether it is retreating or advancing, and is thus an essential feature to monitor in the context of climate change. Furthermore, it is visible in satellite imagery and can thus be observed from space. In this project, we will apply Deep Learning techniques to try to map the calving front of [23 Greenland and 2 Antarctic outlet glaciers](https://opara.zih.tu-dresden.de/xmlui/handle/123456789/5721).

**Challenge**: Since the glacier dataset is too small to provide the training for a robust model, we will apply transfer learning of a model pretrained on coastline mapping, since it leads to a similar segmentation problem on much larger available datasets.

### Data
