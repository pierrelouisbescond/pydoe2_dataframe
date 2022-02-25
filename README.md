# pydoe2_dataframe

This project aims to extend the functionalities from pyDOE2 by allowing its user to provide a Pandas DataFrame including the different features' levels:

|Feat1|Feat2|
|--|--|
|0|100|
|1|200|

and get, in return, the corresponding DoE plan :

    create_fact_plan(data_2x2_in)

|Feat1|Feat2|
|--|--|
|0|100|
|0|200|
|1|100|
|1|200|

The script offers the possibility of choose different options:

 - **Full-factorial plan** or **Generalized Subset Designs** (method_choice)
 - **Replication** of the DoE (replication_factor)
 - **Randomization** of the DoE (randomization)

Example with a 3 x 2 DataFrame:
|Feat1|Feat2|Feat3|
|--|--|--|
|0|10|50|
|1|20|60|

    create_fact_plan(data_3x2_in, randomization=True)

|Feat1|Feat2|Feat3|
|--|--|--|
|0|20|60|
|1|20|50|
|0|10|50|
|1|10|60|
|1|10|50|
|0|10|60|
|1|20|60|
|0|20|50|


    create_fact_plan(data_3x2_in, randomization=True, method_choice="gsd", reduction_ratio=2)

|Feat1|Feat2|Feat3|
|--|--|--|
|0|10|50|
|0|20|60|
|1|10|60|
|1|20|50|