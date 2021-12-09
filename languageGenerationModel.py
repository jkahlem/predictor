from simpletransformers.t5 import T5Model, T5Args
from transformers.training_args import TrainingArguments
import model
from util import copyTo
from uuid import uuid4
import pandas as pd
from config import get_model_config, get_resource_path, get_script_dir, is_cuda_available
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.trainer import Trainer
from transformers.models.auto import AutoModel
import torch
import math

# TODO: rework model loading... (method generation needs two models - language modeling model and generator, but model holder supports only one model)
#       -> the underlying model should do the needed stuff. However, the model holder should call the loading/initializing methods instead of the message script.
#       -> maybe rename to "initialize/load for training" and "initialize/load for prediction"...
#       - Also: clear temporary files (train/eval messages ...)
#
#  Using current T5 model works pretty good! example predictions:
#  - "getName" : "void."
#  - "createPerson" : "void."
#  - "setPosition" : "position."
#  - "withName" : "name."
#  - "findByNameOrAge" : "name, age."
#  - "compare" : "a, b."

class MethodGenerationModel(model.Model):
    def __init__(self):
        self.dataset = None
        self.languageModeling = None
        self.languageGenerator = None

    # prints a message if cuda is used or not
    def __print_model_initialization(self) -> None:
        if is_cuda_available():
            print('Initialize language modelling model using CUDA')
        else:
            print('Initialize language modelling model without CUDA')
    
    def __languageModelingArgs(self) -> LanguageModelingArgs:
        return LanguageModelingArgs(cache_dir=self.cacheDirName(), output_dir=self.outputsDirName(), mlm=False)

    def __languageGenerationArgs(self) -> LanguageGenerationArgs:
        return LanguageGenerationArgs(cache_dir=self.cacheDirName(), output_dir=self.outputsDirName())

    # initializes a new model
    def init_new_model(self) -> None:
        self.__print_model_initialization()
        used_model_type, used_model = get_model_config()
        #self.languageModeling = LanguageModelingModel('gpt2', 'gpt2', use_cuda=is_cuda_available(), args=self.__languageModelingArgs())

    # Loads an already created/trained classification model
    def load_model(self) -> None:
        pass
        #self.languageGenerator = LanguageGenerationModel('gpt2', self.outputsDirName(), use_cuda=is_cuda_available(), args=self.__languageGenerationArgs())

    # https://huggingface.co/blog/how-to-generate
    # https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0
    # https://link.springer.com/chapter/10.1007/978-3-030-91699-2_15
    # https://ieeexplore.ieee.org/document/9402114
    # https://jaketae.github.io/study/keyword-extraction/
    # https://www.lighttag.io/blog/sequence-labeling-with-transformers/
    # github.com/abhimishra91/transformers-tutorials/

    # Trains the model using the given training set
    def train_model(self, training_set: str) -> None:
        #tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
        #trainer = WordPieceTrainer(special_tokens=['[UNK]','[CLS]','[SEP]','[PAD]','[MASK]','[MTD]','[/MTD]','[TYP]','[/TYP]','[PAR]','[/PAR]'])
        #tokenizer.pre_tokenizer = Whitespace()
        #tokenizer.train([training_set], trainer)
        #tokenizer.save('customTokenizer.json')
        #inputs = tokenizer(training_set, padding='max_length', truncation=True)
        #m = AutoModel.from_pretrained('bert-base-uncased')
        #args = TrainingArguments("test")
        #t = Trainer(model=m, args=args, train_dataset=inputs)
        #t.train()
        #t.save_state()
        '''tokenizer = Tokenizer(WordPiece())
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_special_tokens(['<MSP>', '<PSP>'])
        tokenizer.train(files=[training_set])
        tokenizer.save('customTokenizer.json')
        mappedTokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        m = AutoModel.from_pretrained('bert-base-uncased')
        args = TrainingArguments('test')
        t = Trainer(model=m, args=args, tokenizer=mappedTokenizer, train_dataset=mappedTokenizer(self.dataset, padding='max_length', truncation=True))
        t.train()
        t.save_state()'''
        '''
        trainDf = pd.read_csv(training_set, header=None, names=['input_text', 'target_text'], sep=';', na_filter=False)
        args = Seq2SeqArgs(num_train_epochs=5)
        model = Seq2SeqModel('roberta', 'roberta-base', 'bert-base-uncased', args=args)
        model.train_model(trainDf)
        '''
        '''
        print('Load sequences:\n')
        trainDf = pd.read_csv(training_set, header=None, names=['input_text', 'target_text'], sep=';', na_filter=False)
        input_sequences = list()
        output_sequences = list()
        task_prefix = "get parameters for: "
        for i, row in trainDf.iterrows():
            input_sequences.append(task_prefix + row[0])
            output_sequences.append(row[1])

        #self.languageModeling.train_model(training_set)
        print('Tokenize sequences:\n')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        encoding = tokenizer(input_sequences, padding='longest', max_length=64, truncation=True, return_tensors='pt')
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        #encode targets
        target_encodings = tokenizer(output_sequences, padding='longest', max_length=128, truncation=True)

        print('Create labels:\n')
        labels = target_encodings.input_ids

        labels = [
            [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in labels
        ]
        labels = torch.tensor(labels)

        rows = len(input_ids)
        rowsPerStep = 64
        steps = 50 #math.floor(rows / rowsPerStep + 1) 
        print('Rows ' + str(rows) + ' / rows per step ' + str(rowsPerStep) + ' / steps ' + str(steps))
        for i in range(steps):
            print('Train model: Step #' + str(i))
            inputs, attentions, ls = list(), list(), list()
            if i*rowsPerStep >= rows:
                break
            elif steps*rowsPerStep >= rows and i == steps-1:
                inputs, attentions, ls = input_ids[rowsPerStep*i:], attention_mask[rowsPerStep*i:], labels[rowsPerStep*i:]
            else:
                n = i+1
                inputs, attentions, ls = input_ids[rowsPerStep*i:rowsPerStep*n], attention_mask[rowsPerStep*i:rowsPerStep*n], labels[rowsPerStep*i:rowsPerStep*n]
            loss = model(input_ids=inputs, attention_mask=attentions, labels=ls).loss
            print('Loss:\n' + str(loss))

        print('Generate outputs:')
        testMethod = task_prefix + 'find person by name'
        testinputids = tokenizer(testMethod, return_tensors='pt').input_ids
        outputs = model.generate(testinputids)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
        tokenizer.save_pretrained(get_resource_path('customTokenizer'))
        model.save_pretrained(get_resource_path('outputs'))
        '''
        trainDf = pd.read_csv(training_set, header=None, names=['prefix', 'input_text', 'target_text'], sep=';', na_filter=False)
        args = T5Args()
        args.num_train_epochs = 3
        model = T5Model('t5', 't5-small', args=args, use_cuda=True)
        model.train_model(trainDf)
        

    # Evaluates the model using the given evaluation set
    def eval_model(self, evaluation_set: list) -> dict:
        pass
        #return self.languageModeling.eval_model(evaluation_set)

    # Not needed for parameter generation
    def load_additional(self, additional_data) -> None:
        pass

    # Makes predictions
    def predict(self, predictionData: list) -> list:
        '''print('load tokenizer / mmodel:\n')
        task_prefix = "get parameters for: "
        tokenizer = T5Tokenizer.from_pretrained(get_resource_path('customTokenizer'))
        model = T5ForConditionalGeneration.from_pretrained(get_resource_path('outputs'))
        output = list()
        for methodName in predictionData:
            print("Generate for '"  + methodName + "'")
            inputs = tokenizer(task_prefix + methodName, return_tensors='pt')
            outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            #generated = self.languageGenerator.generate(methodName)[0]
            print('generated: ' + str(tokenizer.decode(outputs[0])))
            output.append(methodName)
            '''
        model = T5Model('t5', 'outputs')
        inputs = list()
        for methodName in predictionData:
            inputs.append("generate parameters: " + methodName)
        outputs = model.predict(inputs)
        print(outputs)
        return outputs

    # path addition for the cacheDir
    def cacheDirName(self) -> str:
        return 'cache_dir/'

    # path addition for the outputs dir 
    def outputsDirName(self) -> str:
        return 'outputs/'

    # formats input data
    def dataFormatter(self, data):
        filename = 'lmd_' + str(uuid4())
        filepath = get_resource_path(filename)
        copyTo(data, filepath)
        
        formatted = list()
        for line in data.readlines():
            if (line[len(line)-1]) == '\n':
                line = line[:len(line)-1]
            formatted.append(line[:len(line)-2])
        self.dataset = formatted
        return filepath
        #formatted = list()
        #for line in data.readlines():
        #    if (line[len(line)-1]) == '\n':
        #        line = line[:len(line)-1]
        #    formatted.append(line[:len(line)-2])
        #return formatted
