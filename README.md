# pydoe2_dataframe

This project aims to extend the functionalities from pyDOE2 by allowing its user to provide a Pandas DataFrame including the different features' levels:

|Feat1|Feat2|
|--|--|
|0|100|
|1|200|

and get, in return, the corresponding DoE plan :
|Feat1|Feat2|
|--|--|
|0|100|
|0|200|
|1|100|
|1|200|

with the possibility to choose (at the moment), a full-factorial plan or Generalized Subset Designs.
