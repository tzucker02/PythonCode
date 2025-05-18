The below table is the data dictionary for a sanitized version of the titanic dataset. An Excel version of this file is available [here](https://github.com/tzucker02/602_final_project/raw/main/titanic%20dataset%20preliminary%20dictionary.xlsx).

| Feature | Description | Possible Values | Count/Distribution of # in each category |
|----|----|----|----|
| gender | Sex of individual | This can be represented either as | |
| |  | 0 = Male, or 1 = Female | 0 - 1635 or 1 - 464 |
| age       | Age of passenger | continuous type - top 5 reported | 22 - 95 |
| | | | 30 - 87 |
| | | | 24 - 81 |
| | | | 28 - 79 |
| | | | 25 - 78 |
| class    | Cabin Class of passenger | 0=3rd class | 649 |
| | | 1=2nd class | 263 |
| | | 2=1st class | 301 |
| | | 3=Engineering crew | 322 |
| | | 4=Victualling crew | 429 |
| | | 5=Restaurant staff | 69 |
| | | 6=Deck crew | 66 |
| embarked | Port that the passenger embarked from | 0=Southhampton | 1545 |
| | | 1=Cherbourg | 244 |
| | | 2=B | 188 |
| | | 3=Queenstown | 122 |
| country  | Country - note that countries with fewer than 20 entries are put together in the other category | 00 - England | 00 - 1113 |
| | | 01 - United States |   01 - 262 |
| | | 02 - Ireland |  13 - 182 |
| | | 03 - Sweden | 02 - 136 |
| | | 04 - Lebanon |   03 - 104 |
| | | 05 - Finland | 04 - 71 |
| | | 06 - Scotland | 05 - 54 |
| | | 07 - Canada | 06 - 35 |
| | | 08 - Norway | 07 - 34 |
| | | 09 - France | 08 - 26 |
| | | 10 - Belgium | 09 - 25 |
| | | 11 - Northern Ireland | 10 - 22 |
| | | 12 - Wales | 12 - 20 |
| | | 13 - Other | 11 - 15 |
| fare | continuous field | top 5 fares reported in dataset | 0 - 886 |
| | | | 8.01 - 57 |
| | | | 13 - 57 |
| | | | 7.15 - 54 |
| | | | 26 - 45 |
| sibsp  | Siblings and/or Spouses | 0, 1, 2, 3, 4, 5, 8 | 0 - 1693 |
| | | | 1 - 311 |
| | | | 2 - 42 |
| | | | 4 - 22 |
| | | | 3 - 16 |
| | | | 8 - 9 |
| | | | 5 - 6 |
| parch | Parents and/or Children | 0, 1, 2, 3, 4, 5, 6, 9 | 0 - 1805 |
| | | | 1 - 160 |
| | | | 2 - 111 |
| | | | 3 - 8 |
| | | | 5 - 6 |
| | | | 4 - 5 |
| | | | 6 - 2 |
| | | | 9 - 2 |
| target | Survived | 0 - No, 1 - Yes | 0 - 1418 |
| | | | 1 - 681 |
|  Further analysis might derive this information:  | Did the person travel with or without family? <b>(Alone_or_family)</b></br><br>This is derived from the two fields sibsp (siblings and/or spouses) and parch (parents or children). This was derived by adding the sibsp and parch fields and testing if this combined number is greater than 0. If greater than 0, the passenger is part of a family. Otherwise, the passenger is considered to be travelling alone.  | 0 = Alone<br>1 = family | 0 - 1597 |
| | | | 1 - 502 |
