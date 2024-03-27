# HierarT-main

## Operational Guidelines
If you want to execute HierarT using ICEWS14 as an example, you need to generate the data required for model execution first (see Step 1); 
the command for model training can be found in Step 2; 
and the command for model testing is provided in Step 3.
1. ready for training:
```shell
unzip data.zip
python preprpcess_data.py --data_dir data/ICEWS14 --store_options_num 200 --store_actions_num 50
python text_transformer.py --data_path data/ICEWS14
```

2. train model:
```shell
python main.py --do_train --cuda --text_transformer --kmeans_reward_shaping --data_path data/ICEWS14 --max_option_num 200 --max_action_num 100
```

3. text model:
```shell
python main.py --cuda --do_test --IM --data_path data/ICEWS14 --load_model_path xxxxx
```

## Hyperparameter Settings

### ICEWS14

|  Hyperparameter   | Value  |
|  ----  | ----  |
| batch_size  | 512 |
| path_length  | 2 |
| learning rate of relation level  | 0.0003 |
| learning rate of entity level  | 0.0001 |
| dropout rate 0 of relation level  | 0.3 |
| dropout rate 0 of entity level  | 0.2 |
| entity cluster size  | 4 |
| time cluster size  | 6 |
| relation level beam size  | 10 |
| entity level beam size  | 60 |
| beam size  | 80 |
|store options num | 200 |
|store actions num | 50 |
|max option num | 200 |
|max action num | 100 |

### ICEWS18

|  Hyperparameter   | Value  |
|  ----  | ----  |
| batch_size  | 256 |
| path_length  | 2 |
| learning rate of relation level  | 0.0001 |
| learning rate of entity level  | 0.0001 |
| dropout rate 0 of relation level  | 0.3 |
| dropout rate 0 of entity level  | 0.2 |
| entity cluster size  | 6 |
| time cluster size  | 5 |
| relation level beam size  | 10 |
| entity level beam size  | 40 |
| beam size  | 60 |
|store options num | 200 |
|store actions num | 50 |
|max option num | 200 |
|max action num | 100 |

## WIKI

|  Hyperparameter   | Value  |
|  ----  | ----  |
| batch_size  | 256 |
| path_length  | 2 |
| learning rate of relation level  | 0.0001 |
| learning rate of entity level  | 0.0001 |
| dropout rate 0 of relation level  | 0.1 |
| dropout rate 0 of entity level  | 0.1 |
| entity cluster size  | 7 |
| time cluster size  | 6 |
| relation level beam size  | 10 |
| entity level beam size  | 40 |
| beam size  | 60 |
|store options num | 50 |
|store actions num | 600 |
|max option num | 100 |
|max action num | 200 |

## YAGO

|  Hyperparameter   | Value  |
|  ----  | ----  |
| batch_size  | 128 |
| path_length  | 4 |
| learning rate of relation level  | 0.0001 |
| learning rate of entity level  | 0.0001 |
| dropout rate 0 of relation level  | 0.1 |
| dropout rate 0 of entity level  | 0.1 |
| entity cluster size  | 7 |
| time cluster size  | 6 |
| relation level beam size  | 10 |
| entity level beam size  | 60 |
| beam size  | 80 |
|store options num | None |
|store actions num | None |
|max option num | 100 |
|max action num | 200 |

