# pyrefly: ignore [missing-import]
from datacollection.dataset_builder import DatasetBuilder

dataset_builder = DatasetBuilder()
x, y = dataset_builder.build_dataset()
print(x.shape)
print(y.shape)

print(y)    