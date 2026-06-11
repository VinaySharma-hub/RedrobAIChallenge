Copy these files into REDROB_AI_CHALLENGE/Agents_work/:
- config.py
- agents_dev.py
- ranker.py
- run.py
- requirements.txt

Create folders:
REDROB_AI_CHALLENGE/data/
REDROB_AI_CHALLENGE/outputs/

Put your files:
data/candidates.jsonl
data/job_description.txt

Run:
cd C:\Users\Welcome\OneDrive\Dokumen\REDROB_AI_CHALLENGE
venv\Scripts\activate
pip install -r Agents_work\requirements.txt
python Agents_work\run.py

Must print:
USING CORRECTED RANKER V3 - NESTED REDROB SCHEMA ENABLED

The debug variation should be:
behavior unique > 1
experience unique > 1
title unique > 1
