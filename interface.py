from tkinter import *
from keras.models import model_from_json
import pandas as pd
from sklearn import preprocessing

fields = ('Permit Type',
          'Permit Type Definition',
          'Street Name',
          'Street Suffix',
          'Current Status',
          'Current Status Date',
          'Structural Notification',
          'Number of Existing Stories',
          'Number of Proposed Stories',
          'Voluntary Soft-Story Retrofit',
          'Fire Only Permit',
          'Revised Cost',
          'Proposed Use',
          'Proposed Units',
          'TIDF Compliance',
          'Existing Construction Type',
          'Existing Construction Type Description',
          'Proposed Construction Type',
          'Proposed Construction Type Description',
          'Supervisor District',
          'Neighborhoods - Analysis Boundaries',
          'Zipcode')


def predict(entries):


    model = import_model()

    val1 = entries['Permit Type'].get()
    val2 = entries['Permit Type Definition'].get()
    val3 = entries['Street Name'].get()
    val4 = entries['Street Suffix'].get()
    val5 = entries['Current Status'].get()
    val6 = entries['Current Status Date'].get()
    val7 = entries['Structural Notification'].get()
    val8 = entries['Number of Existing Stories'].get()
    val9 = entries['Number of Proposed Stories'].get()
    val10 = entries['Voluntary Soft-Story Retrofit'].get()
    val11 = entries['Fire Only Permit'].get()
    val12 = entries['Revised Cost'].get()
    val13 = entries['Proposed Use'].get()
    val14 = entries['Proposed Units'].get()
    val15 = entries['TIDF Compliance'].get()
    val16 = entries['Existing Construction Type'].get()
    val17 = entries['Existing Construction Type Description'].get()
    val18 = entries['Proposed Construction Type'].get()
    val19 = entries['Proposed Construction Type Description'].get()
    val20 = entries['Supervisor District'].get()
    val21 = entries['Neighborhoods - Analysis Boundaries'].get()
    val22 = entries['Zipcode'].get()

    df = pd.DataFrame([val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,val21,val22])
    print(df)



    min_max_scalerX2 = preprocessing.MinMaxScaler()
    X_scaled2 = min_max_scalerX2.fit_transform(df)




def makeform(root, fields):

    entries = {}

    for field in fields:
        row = Frame(root)
        lab = Label(row, width=22, text=field+": ", anchor='w')
        ent = Entry(row)
        ent.insert(0,"0")
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries[field] = ent

    return entries




def import_model():


    # load json and create model
    json_file = open('model\delaypermit.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    classifier = model_from_json(loaded_model_json)
    # load weights into new model
    classifier.load_weights("model\delaypermit.h5")
    print("Loaded model from disk")

    return classifier



if __name__ == '__main__':
    root = Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = Button(root, text='Predict Delay',command=(lambda e=ents: predict(e)))
    b1.pack(side=LEFT, padx=5, pady=5)
    b3 = Button(root, text='Quit', command=root.quit)
    b3.pack(side=LEFT, padx=5, pady=5)
    root.mainloop()