
# CKIP Dependency Parsing Demo

DEMO Website: https://ckip.iis.sinica.edu.tw/service/dependency-parser/

# Model Download

Drive link: ###
```
mkdir demo_model
```

# Run CKIP tagger server

Download ```./data``` from https://github.com/ckiplab/ckiptagger (CkipTagger)

```Run in tmux```
```
cd tagger/

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 server.py --port 5555
```

# Run demo
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run demo.py --server.fileWatcherType none
```