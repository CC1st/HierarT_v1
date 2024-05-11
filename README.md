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


|  Hyperparameter  | Value (ICEWS14) | Value (ICEWS18) | Value (WIKI)  | Value (YAGO)  |
|  ----  |  ----  | ----  |  ----  | ----  |
| batch_size  | 512 |256 | 256 | 128 |
| path_length  | 2 | 2 | 2 | 4 |
| learning rate of relation level  | 0.0003 | 0.0001 | 0.0001 | 0.0001 |
| learning rate of entity level  | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| dropout rate 0 of relation level  | 0.3 | 0.3 | 0.1 | 0.1 |  
| dropout rate 0 of entity level  | 0.2 | 0.2 | 0.1 | 0.1 |
| entity cluster size  | 4 | 6 | 7 | 7 |
| time cluster size  | 6 | 5 | 6 | 6 |
| relation level beam size  | 10 | 10 | 10 | 10 |
| entity level beam size  | 60 | 40 | 60 | 60 |
| beam size  | 80 | 60 | 80 | 80 |
|store options num | 200 | 200 | 50 | None |
|store actions num | 50 | 50 | 600 | None |
|max option num | 200 | 200 | 100 | 100 |
|max action num | 100 | 100 | 200 | 200 |

