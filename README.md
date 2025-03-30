# SyllabusQA

This repo contains the data and code for the paper <a href="https://aclanthology.org/2024.acl-long.557/">SyllabusQA: A Course Logistics Question Answering Dataset</a>.

If you use our data or code, or find this work helpful for your research, then please cite us!
```
@inproceedings{fernandez-etal-2024-syllabusqa,
    title = "{S}yllabus{QA}: A Course Logistics Question Answering Dataset",
    author = "Fernandez, Nigel  and
      Scarlatos, Alexander  and
      Lan, Andrew",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.557/",
    doi = "10.18653/v1/2024.acl-long.557",
    pages = "10344--10369"
}
```

## Data

TODO

## Code

### Setup

Create Python environment:
```
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

### Benchmarks

TODO

### Evaluation

To evaluate answer quality, we employ a novel factuality metric (Fact-QA) and textual similarity metrics (BERTScore and ROUGE).

Note that Fact-QA requires querying OpenAI, so you'll need to set an API key:
```
export OPENAI_API_KEY=<your api key>
```

Run the following to evaluate the generated answers:
```
cd metric
python eval_results.py ../data/dataset_split/test.csv <your output file> --metrics --nrows <number of rows to evaluate> --model <openai model name> --out <path to output result file (csv)>
```

## Code Contributors
Nigel Fernandez\
Alexander Scarlatos