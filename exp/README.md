

### Steps to Run Experiments

#### 1. Experiments on UT-Zappos Dataset
 - Run model training:
    ```bash
    # suppose you are under the current folder ./exp in a terminal session (screen/tmux etc).
    cd ut-zappos
    bash train_model.sh 3 ctdl_t5_t64_i8 gencsp | tee train_ctdl_t5_t64_i8.log
    ```
    - Training results will be in the folder `output/ut-zappos/ctdl_t5_t64_i8/`

 - Run model evaluation (after the model finished training, or val AUC is unlikely to further increase):

    - Go the the folder `output/ut-zappos/ctdl_t5_t64_i8/checkpoints/` and find out which historical (NOT the latest) checkpoint remained to be the best, e.g., `model_12.pt`.
    - Make sure the `config/ut-zappos/ctdl_t5_t64_i8/eval.yml` use the best checkpoint.
    - Run **Closed-World** evaluation (finish in 2-3 mins):
        ```bash
        bash eval_model.sh 3 closed ctdl_t5_t64_i8 gencsp | tee ../../output/ut-zappos/ctdl_t5_t64_i8/eval_closed_e12.log
        ```
    - Run **Open-World** evaluation (may be more than 10 mins):
        ```bash
        bash eval_model.sh 3 open ctdl_t5_t64_i8 gencsp | tee ../../output/ut-zappos/ctdl_t5_t64_i8/eval_open_e12.log
        ```
    - Evaluation results will be in the folder `output/ut-zappos/ctdl_t5_t64_i8/`

#### 2. Experiments on C-GQA Dataset
 - Run model training:
    ```bash
    # suppose you are under the current folder ./exp in a terminal session (screen/tmux etc).
    cd cgqa
    bash train_model.sh 3 ctdl_t5_t64_i8 gencsp | tee train_ctdl_t5_t64_i8.log
    ```
    - Training results will be in the folder `output/cgqa/ctdl_t5_t64_i8/`

 - Run model evaluation (after the model finished training, or val AUC is unlikely to further increase):

    - Go the the folder `output/cgqa/ctdl_t5_t64_i8/checkpoints/` and find out which historical (NOT the latest) checkpoint remained to be the best, e.g., `model_12.pt`.
    - Make sure the `config/cgqa/ctdl_t5_t64_i8/eval.yml` use the best checkpoint.
    - Run **Closed-World** evaluation (finish in 2-3 mins):
        ```bash
        bash eval_model.sh 3 closed ctdl_t5_t64_i8 gencsp | tee ../../output/cgqa/ctdl_t5_t64_i8/eval_closed_e12.log
        ```
    - Run **Open-World** evaluation (may be more than 10 mins):
        ```bash
        bash eval_model.sh 3 open ctdl_t5_t64_i8 gencsp | tee ../../output/cgqa/ctdl_t5_t64_i8/eval_open_e12.log
        ```
    - Evaluation results will be in the folder `output/cgqa/ctdl_t5_t64_i8/`

#### 3. If you want to modify some configs
- Copy a config folder, e.g., `cd config/cgqa/ & cp ctdl_t5_t64_i8 mycfg`.
- Change your parameters of `mycfg` in both `eval.yml` and `train.yml`, make sure that these two yml file contains correct `save_path` and `ckpt_file`, respectively:
    ```python
        save_path: output/cgqa/mycfg
        ckpt_file: output/cgqa/mycfg/checkpoints/model_12.pt
    ```
- Run `train_model.sh` and `eval_model.sh` following the step 1 and 2 above.