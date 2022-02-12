from os.path import join
# from IPython.display import HTML, display
import pandas as pd
from transformers import TrainingArguments, Trainer, IntervalStrategy, Adafactor

from src.core.persistance import validate_path


def train(args, model, datasets, mode="torch"):
    if "torch" == mode:
        return train_with_torch(args, model, datasets)
    elif "huggingface" == mode:
        return train_with_huggingface(args, model, datasets)


def train_with_torch(args, model, datasets):
    # Set the model in training mode
    model.train()
    optimizer = optimizer_adafactor(model)
    num_batches = len(datasets['train'])

    loss_per_10_steps = []
    for epoch in range(1, args["num_train_epochs"] + 1):
        print('Running epoch: {}'.format(epoch))
        running_loss = 0

        # out = display(progress(1, num_batches + 1), display_id=True)
        for i in range(num_batches):

            # clear out the gradients of all Variables
            optimizer.zero_grad()
            inputs_embeds = model.embed_tokens(datasets['train']['input_ids'][i])
            attention_mask = model.extend_attention_mask(datasets['train']['attention_mask'][i])
            labels = model.extend_labels(datasets['train']['labels'][i])

            # Forward propagation
            outputs = model(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask)

            loss = outputs.loss
            loss_num = loss.item()
            running_loss += loss_num

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

            if i % 10 == 0:
                loss_per_10_steps.append(loss_num)

            if i % args['logging_steps'] == 0:
                print('Steps: {} , loss: {}'.format(i, loss_num))
                _log(args["output_dir"], "loss_by_steps_" + args['starting_timestamp'] + ".csv", i, loss_num)

        running_loss = running_loss / int(num_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))
        _log(args["output_dir"], "epoch_average_loss_" + args['starting_timestamp'] + ".csv", epoch, running_loss)

    return model


def _log(path, file, index, running_loss):
    path = validate_path(path)
    path = validate_path(join(path, "logs"))
    file = join(path, file)
    df = pd.DataFrame([[index, running_loss]], columns=['index', 'loss'])
    df.to_csv(file, sep='\t', encoding='utf-8', mode='a', header=False, index=False)


def train_with_huggingface(args, model, datasets):
    output_dir = args["output_dir"]
    training_args = TrainingArguments(
        output_dir=join(output_dir, "logs"),
        num_train_epochs=args["num_train_epochs"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["eval_batch_size"],
        eval_accumulation_steps=args["eval_accumulation_steps"],
        prediction_loss_only=args["prediction_loss_only"],
        learning_rate=args["learning_rate"],
        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=args["save_steps"],
        save_total_limit=args["save_total_limit"],
        remove_unused_columns=args["remove_unused_columns"],
        run_name=args["wandb_run_name"],
        logging_steps=args["logging_steps"],
        eval_steps=args["eval_steps"],
        logging_first_step=args["logging_first_step"],
        load_best_model_at_end=args["LOAD_BEST_MODEL_AT_END"],
        metric_for_best_model=args["METRIC_FOR_BEST_MODEL"],
        greater_is_better=args["GREATER_IS_BETTER"]
    )

    model.train()
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
                      eval_dataset=datasets["validation"])

    print("start training")
    trainer.train()
    print("finished training")
    return model


def optimizer_adafactor(model,
                        lr=0.6,  # default values for adafactor
                        eps=(1e-30, 1e-3),  # default values for adafactor
                        clip_threshold=1.0,  # default values for adafactor
                        decay_rate=-0.8,  # default values for adafactor
                        beta1=None,  # default values for adafactor
                        weight_decay=1e-5,  # default values for adafactor
                        relative_step=False,
                        scale_parameter=False,
                        warmup_init=False):
    return Adafactor(
        [model.get_soft_params()],
        lr=lr,
        eps=eps,
        clip_threshold=clip_threshold,
        decay_rate=decay_rate,
        beta1=beta1,
        weight_decay=weight_decay,
        relative_step=relative_step,
        scale_parameter=scale_parameter,
        warmup_init=warmup_init
    )

#
# def progress(loss, value, maximum=100):
#     return HTML(""" Batch loss :{loss}
#         <progress
#             value='{value}'
#             maximum='{maximum}',
#             style='width: 100%'
#         >
#             {value}
#         </progress>
#     """.format(loss=loss, value=value, maximum=maximum))
