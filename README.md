# lang-rep


This is the original implementation of the following paper.

Chi-Liang Liu, Tsung-Yuan Hsu, Yung-Sung Chuang, Chung-Yi Li, and Hung-yi Lee. [Language Representation in Multilingual BERT and its applications to improve Cross-lingual Generalization.](https://arxiv.org/abs/2010.10041)


## Dependency
```
numpy
pytorch
transformers==2.11.0
```

## Extract the language representation (Mean of Langauge)

Put the file you want to extract to `${data_dir}/${langauge}.txt`

For input format, please see `data/en_sample.txt.`
```
./run_extract_wiki_feature.sh $langauge $model_name_or_path $data_dir $block_size
```
It would generate a pickle file: `${langauge}_${model_name_or_path}.pkl` in `./output`.

It should contain a dictionary {'mean': np.array}
np.array should contain the mean of langauge vector from embedding to output layer.

For example, if you use m-BERT as the model. The shape of array should be `(13, 768)`. (embedding + 12 layer, model\_dim)


## Mean Difference Shift or Zero-Mean

Please refer bert.py, if you want to apply on XLM, XLM-R.
The usage is exactly the same as transformers 2.11.0, except there are two extra
arguments: `langauge_layer`, `langauge_hidden_states`

* `langauge_layer`: which layer you want to apply.
* `langauge_hidden_states`: the vector you want add. 
`(for zero-mean, you should feed the -1*(the vector you extracted))`


## Citation
```
@misc{liu2020language,
      title={Language Representation in Multilingual BERT and its applications to improve Cross-lingual Generalization}, 
      author={Chi-Liang Liu and Tsung-Yuan Hsu and Yung-Sung Chuang and Chung-Yi Li and Hung-yi Lee},
      year={2020},
      eprint={2010.10041},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


