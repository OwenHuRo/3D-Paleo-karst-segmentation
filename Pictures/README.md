# Visualizations

`ConstructData/check_Visualization.py` inspects generated seismic/label pairs,
and `Test_Vision_Version.py` compares input, label, probability and binary
prediction volumes.

When requested, `ConstructData/CreateProcess.py` stores the seven intermediate
modeling stages under `Pictures/ModelingProcess/sample_<id>/`. These generated
binary volumes are ignored by Git.

The method diagrams and synthetic comparison used by the main README are copied
from the author-provided manuscript sources. The field-data figure is not
included because the underlying field data are protected by confidentiality
agreements with the data owner.
