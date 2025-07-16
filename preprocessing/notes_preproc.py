#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'MIMIC-IV-Data-Pipeline-main', 'mimiciv', 'notes'))
import os
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule
from spacy.tokens import Span
import spacy
import sys
import pandas as pd
import numpy as np

#Load nlp1 ------
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

#Load nlp2 ------
n_nlp = medspacy.load()

target_matcher = n_nlp.get_pipe("medspacy_target_matcher")
context = n_nlp.get_pipe("medspacy_context")

Span.set_extension("icd_code", default=None)
Span.set_extension('severity', default=None)


#Rules defined
target_rules = [
    TargetRule('cocaine', 'DRUG-USE'),
    TargetRule('heroine', 'DRUG-USE'),
    TargetRule('morphine', 'DRUG-USE'),
    TargetRule('fentanyl', 'DRUG-USE'),

    

    TargetRule('heart failure', 'DIAGNOSIS', attributes={'icd_code': 'I50'}),
    TargetRule('systolic chf', 'DIAGNOSIS', attributes={'icd_code': 'I50.2'}),
    TargetRule('diastolic chf', 'DIAGNOSIS', attributes={'icd_code':'I50.3'}),
    TargetRule('chronic kidney disease', 'DIAGNOSIS',  pattern=[{'LOWER': 'ckd'}], attributes={'icd_code': 'N18'}),
    TargetRule('chronic obstructive pulmonary disease', 'DIAGNOSIS',  pattern=[{'LOWER': 'copd'}], attributes={'icd_code': 'J44'}),
    TargetRule('coronary artery disease', 'DIAGNOSIS', pattern=[{'LOWER': 'cad'}], attributes={'icd_code': 'I25'}),
    

    
    TargetRule('insulin', 'MEDICATION'),
    TargetRule('heparin', 'MEDICATION'),
    TargetRule('furosemide', 'MEDICATION', pattern=[{'LOWER':'lasix'}]),
    TargetRule('norepinephrine', 'MEDICATION'),
    TargetRule('warfarin', 'MEDICATION'),
    TargetRule('metoprolol', 'MEDICATION'),

    TargetRule('shortness of breath', 'SYMPTOM', pattern=[{'LOWER':'sob'}]),
    TargetRule('chest pain', 'SYMPTOM'),
    TargetRule('fever', 'SYMPTOM'),
    TargetRule('abdominal pain', 'SYMPTOM'),
    TargetRule('headache', 'SYMPTOM'),
    TargetRule('fatigue', 'SYMPTOM'),
    TargetRule('diarrhea', 'SYMPTOM'),
    TargetRule('edema', 'SYMPTOM'),
    TargetRule('palpitations', 'SYMPTOM'),
    
    
]


family_rules = [
    ConTextRule("family history", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("mother had", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("father had", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("sister had", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("brother had", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("runs in the family", "FAMILY MEDICAL HISTORY", direction="BACKWARD"),
    ConTextRule("maternal", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("paternal", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("hereditary", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
    ConTextRule("genetic", "FAMILY MEDICAL HISTORY", direction="FORWARD"),
]

severity = [
    ConTextRule('mild', "SEVERITY", direction="FORWARD"),
    ConTextRule('severe', 'SEVERITY', direction="FORWARD"),
]


context_rules = [
    ConTextRule("presenting with", "CURRENT", direction="FORWARD"),
    ConTextRule("past", "HISTORICAL", direction="FORWARD"),
    ConTextRule("history of", "HISTORICAL", direction="FORWARD"),
    ConTextRule("no", "NEGATED", direction="FORWARD"),
    ConTextRule("denies", "NEGATED", direction="FORWARD"),
    ConTextRule('lack', 'NEGATED', direction='FORWARD'),
    ConTextRule('self-discontinuing', 'NEGATED', direction='FORWARD'),
    
    
    ConTextRule(";", "TERMINATE", direction="TERMINATE"),
                ]



#Add rules
context.add(context_rules)
target_matcher.add(target_rules)





def lemmatize(note, nlp):
    '''
    Lemmatize the notes to reduce them to their basic form. 

    Arg:
        notes (str) - a section of the note 
        nlp (medspacy pipe) - applies the pipe and the lemma of it
    Returns:
        str - the section of the note lemmatized
    '''
    
    doc = nlp(note)
    lemNote = [wd.lemma_ for wd in doc]
    return " ".join(lemNote)


def get_csv(sec):
    '''
    Open .csv file from what section the user chose.

    Arg:
        sec (str) - section the user chose
    Returns:
        pandas Dataframe - the opened .csv file
    '''
        

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.abspath(os.path.join(script_dir, '..', 'mimiciv', 'notes'))

    return pd.read_csv(csv_path + '/' + sec.lower() + '.csv.gz', compression='gzip', header=0, index_col=None)



def section_splitting(df):

    df.set_index('note_id', inplace=True)

    data = []

    for i, r_text in enumerate(df['text']):

        note_id = df.index[i]
        doc,_ = lemmatize(r_text, nlp)

        for sent in doc.sents:
            sent_text = str(sent)
            
            if ':' in sent_text.split(' ')[0]:
                section = sent_text.split(':')[0]
                content = sent_text[len(section)+1:].strip()
                data.append([section.strip('\n'), content])


    f = pd.DataFrame(data, columns=['section', 'text'])
    return f

            
def negation_dt(df):

    df.set_index('note_id', inplace=True)

    data = []
    
    for i, r_text in enumerate(df['text']):

        note_id = df.index[i]
        doc = n_nlp(r_text)
    
    
        for ent in doc.ents:

            ent._.is_family = any(mod.category == "FAMILY MEDICAL HISTORY" for mod in ent._.modifiers)

            data.append([
                note_id,
                ent.label_,
                ent.text,
                ent._.is_negated,
                ent._.is_historical,
                ent._.is_family,
                ent._.icd_code
            ])


    f = pd.DataFrame(data, columns=['note_id', 'label', 'ent', 'is_negated', 'is_historical', 'is_family', 'icd_code'])
    return f


def severity_dt(df):

    df.set_index('note_id', inplace=True)

    data = []

    for i, r_text in enumerate(df['text']):

        note_id = df.index[i]
        doc = n_nlp(r_text)

        for ent in doc.ents:

            severity_mod = next((mod for mod in ent._.modifiers if mod.category == "SEVERITY"), None)
            if severity_mod:
                ent._.severity = severity_mod.text

        data.append([
                note_id,
                ent.label_,
                ent.text,
                ent._.is_negated,
                ent._.is_historical,
                ent._.severity,
                ent._.icd_code
            ])
    f = pd.DataFrame(data, columns=['note_id', 'label', 'ent', 'is_negated', 'is_historical', 'severity', 'icd_code'])
    return f

    

    

def extract_data(sec, pred):

    df = get_csv(sec)
    print(f"===========MIMIC-IV Notes============\nExtracting for: {sec}, {pred}")

    match pred:
        case 'Section Splitting':
            return section_splitting(df)

        case 'Negation Detection':
            return negation_dt(df)

        case 'Severity Detection':
            return severity_dt(df)


            

def feature_notes(diag_flag, med_hist, f_med_hist, hist_drug, hist_mi, df, pred):
    selected_labels = []

    if diag_flag:
        selected_labels.append("DIAGNOSIS")
    if med_hist:
        selected_labels.append("MEDICATION")  # or whatever label you use for med history
    if f_med_hist:
        selected_labels.append("FAMILY MEDICAL HISTORY")
    if hist_drug:
        selected_labels.append("DRUG-USE")
    if hist_mi:
        selected_labels.append("MENTAL ILLNESS")  # update this to your actual label if you have one

    if pred == 'Negation Detection' or pred == 'Severity Detection':
        df = df[df['label'].isin(selected_labels)]

    return df
            





    
