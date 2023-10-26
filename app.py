from flask import Flask, request, render_template, jsonify
import pickle, joblib
import numpy as np
import pandas as pd
from flask_wtf import FlaskForm
from wtforms import SelectField

model = pickle.load(open("models/svc_model.pkl", "rb"))
OHE = joblib.load(open("encoders/ohe.joblib", "rb"))
LE = pickle.load(open("encoders/le.pkl", "rb"))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'

input_choices = [
    ['Chrome', 'Electron', 'Firefox', 'Headless', 'Microsoft Edge', 'Opera', 'PhantomJS', 'Safari','NA'],
    ['Android','iOS','Linux', 'Mac','OpenVMS','Solaris','Spark','Ubuntu','Windows','NA'],
    ['Beanshell', 'C#', 'C++', 'Groovy', 'Java', 'Javascript', 'JEXL', 'Kotlin', 'Perl', 'PHP', 'Python','Ruby','Scriptless',
    'SenseTalk', 'Swift','TypeScript', 'VB.Net', 'VBScript'],
    ['API','Database', 'Desktop', 'Mainframe', 'Mobile', 'Web'],
    ['CSV', 'Document', 'Email Report', 'Excel', 'HTML', 'JSON', 'Junit', 'PDF', 'PPT', 'RTF', 'XML', 'xUnit'],
    ['Simple', 'Medium', 'Any of the above'],
    ['Code', 'Low Code', 'No code', 'Any of the above' ],
    ['Large', 'Medium', 'Small', 'Any of the above'],
    ['Yes', 'No', 'Any of the above']
]

browser_os_combination = {
    'Chrome': ['Windows', 'Mac', 'Linux','Spark','OpenVMS'], \
    'Firefox': ['Windows','Mac','Linux','Solaris','Spark','Ubuntu','OpenVMS'], \
    'Microsoft Edge': ['Windows','Mac'], \
    'Safari': ['Mac'], \
    'NA': ['Android','iOS','Linux','Mac','NA','Ubuntu','Windows'], \
    'Opera': ['Windows','Mac','Linux'], \
    'PhantomJS': ['Windows','Mac','Linux'], \
    'Electron': ['Windows','Mac','Linux'], \
    'Headless': ['Windows','Mac','Linux']
}

input_features = ['Web_Browser', 'OS', 'ProgramingLanguage_ToolSupports', 'Application_types', 'Reporting_Feature',
                  'Learning_Curve', \
                  'Tool_Type', 'used_by', 'AI_Enabled']
features_map = {
    'Web_Browser': 'Web Browser', 'OS': 'Operating System', 'ProgramingLanguage_ToolSupports': 'Programming Language',
    'Application_types': 'Application Type', 'Reporting_Feature': 'Reporting Feature',
    'Learning_Curve': 'Learning Curve',
    'Tool_Type': 'Tool Type',
    'used_by': 'Firm Size', 'AI_Enabled': 'AI Enabled'
}

input_choices_map = {input_features[i]: [(choice, choice) for choice in input_choices[i]] for i in
                     range(len(input_features))}

print(input_choices_map)


class Form(FlaskForm):
    Web_Browser = SelectField('Web_Browser', choices=input_choices_map['Web_Browser'])
    OS = SelectField('OS', choices=input_choices_map['OS'])
    ProgramingLanguage_ToolSupports = SelectField('ProgramingLanguage_ToolSupports',
                                                  choices=input_choices_map['ProgramingLanguage_ToolSupports'])
    Application_types = SelectField('Application_types', choices=input_choices_map['Application_types'])
    Reporting_Feature = SelectField('Reporting_Feature', choices=input_choices_map['Reporting_Feature'])
    Learning_Curve = SelectField('Learning_Curve', choices=input_choices_map['Learning_Curve'])
    Tool_Type = SelectField('Tool_Type', choices=input_choices_map['Tool_Type'])
    used_by = SelectField('used_by', choices=input_choices_map['used_by'])
    AI_Enabled = SelectField('AI_Enabled', choices=input_choices_map['AI_Enabled'])


@app.route('/')
def index():
    form = Form()
    return render_template('index.html', form=form)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_values_map = {feature: request.form.get(feature) for feature in input_features}


        input_values = np.array([[value.lower() for key, value in input_values_map.items()]]).reshape(1, -1)
        input_df = pd.DataFrame(input_values, columns=input_features)
        input_encoded = OHE.transform(input_df)
        output_value = LE.inverse_transform(model.predict(input_encoded))[0]

        return render_template("result.html", result=output_value, input_values_map=input_values_map,
                               features_map=features_map)

    return render_template('index.html')


@app.route('/os/<web_browser>')
def getOperatingSystem(web_browser):
    OS_list = browser_os_combination[web_browser]

    OSArray = []

    for opSys in OS_list:
        OSObj = {}
        OSObj['id'] = opSys
        OSObj['name'] = opSys
        OSArray.append(OSObj)

    return jsonify({'OperatingSystems': OSArray})



if __name__ == '__main__':
    app.run(debug=True)
