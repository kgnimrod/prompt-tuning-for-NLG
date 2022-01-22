import unittest
from os.path import join

import torch
import pandas as pd
from transformers import T5Tokenizer

from src.t5_promt_tuning import T5PromptTuning
from src.inference import make_predictions
from src.pre_process import create_list_of_batches


class SimpleSoftPromptInferenceTest(unittest.TestCase):
    def test_infere_on_amr(self):
        load_no_duplicate_sets = True

        # Load the datasets for the Abstract Meaning Representation AMR challenge
        train_data_amr = pd.read_csv(
            join('.', 't5-tuning', 'data', 'amr', 'abstract_meaning_representation.csv')
            if not load_no_duplicate_sets
            else join('.', 't5-tuning', 'data', 'amr', 'train', 'abstract_meaning_representation_train.csv')
        )
        test_data_amr = pd.read_csv(
            join('.', 't5-tuning', 'data', 'amr', 'abstract_meaning_representation.csv')
            if not load_no_duplicate_sets
            else join('.', 't5-tuning', 'data', 'amr', 'test', 'abstract_meaning_representation_test.csv')
        )

        train_data_amr = train_data_amr.sort_values(by='input_text', ignore_index=True)
        test_data_amr = test_data_amr.sort_values(by='input_text', ignore_index=True)
        train_data_amr = train_data_amr.iloc[:len(train_data_amr) - 4, :] \
            if not load_no_duplicate_sets else train_data_amr.iloc[:len(train_data_amr) - 6, :]
        test_data_amr = test_data_amr.iloc[:len(test_data_amr) - 6, :] \
            if not load_no_duplicate_sets else test_data_amr.iloc[:len(test_data_amr) - 3, :]

        batch_size_amr = 8
        number_of_batches_train_amr = int(len(train_data_amr) / batch_size_amr)
        number_of_batches_test_amr = int(len(test_data_amr) / batch_size_amr)

        tokenizer_t5_small = T5Tokenizer.from_pretrained('t5-small')

        number_prompt_tokens = 20
        random_range = 0.5
        init_from_vocab = False

        model_t5_small = T5PromptTuning.from_pretrained(
            't5-small', number_tokens=number_prompt_tokens, initialize_from_vocab=init_from_vocab
        )

        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            dev = torch.device("cpu")
            print("Running on the CPU")
        model_t5_small.to(dev)

        inputs_train_amr, labels_train_amr = create_list_of_batches(batch_size=batch_size_amr,
                                                                    num_batches=number_of_batches_train_amr,
                                                                    data=train_data_amr,
                                                                    tokenizer=tokenizer_t5_small,
                                                                    device=dev)

        encoding_test_amr, target_encoding_test_amr = create_list_of_batches(
            batch_size=batch_size_amr,
            num_batches=number_of_batches_test_amr,
            data=test_data_amr,
            tokenizer=tokenizer_t5_small,
            device=dev
        )
        # train the model...

        model_predictions = make_predictions(model=model_t5_small,
                                             encoding=encoding_test_amr,
                                             tokenizer=tokenizer_t5_small,
                                             )

        print(model_predictions[i] for i in range(len(model_predictions)))

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
