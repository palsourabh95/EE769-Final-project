
from  tkinter import *
from tkinter.tix import *
import pandas as pd
import numpy as np
from tkinter import filedialog
import csv
from xlrd import open_workbook, XLRDError
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def browseFile(btn):
	global address
	global data
	global header
	global dictionary
	dictionary={}

	try:
		if(filetype.get()):
			if(filetype.get()==1):
				root.filename =  filedialog.askopenfilename()
				print(root.filename)
				csv_fileh = open(root.filename, 'r',encoding="ISO-8859-1")

				try:
				    dialect = csv.Sniffer().sniff(csv_fileh.readline())
				    csv_fileh.seek(100)
				except csv.Error:
					label1.config(text="Incompatible format...Choose your file again",relief=RIDGE)
					print(address)
				else:	
					address = root.filename
					label1.config(text="File Address is:   "+address,relief=RIDGE)

					data = pd.read_csv(root.filename,encoding="ISO-8859-1")
					data = data.replace(np.nan, 0)
					data = data.replace(np.inf, 0)


					with open(root.filename, "r",encoding="ISO-8859-1") as f:
					    reader = csv.reader(f)
					    header = next(reader)
					i=1
					for each in header:
						dictionary[i]=each
						i=i+1
 
					btn.config(state="active")



			#Excel read		
			else:
				root.filename =  filedialog.askopenfilename()
				print(root.filename)
				try:
					open_workbook(root.filename,'r')
				except XLRDError:
					label1.config(text="Incompatible format...Choose your file again",relief=RIDGE)
					print(address)
				else:
					address = root.filename
					label1.config(text="File Address is:   "+address,relief=RIDGE)
					data = pd.read_excel(root.filename, sheet_name=None)
					data = data.replace(np.nan, 0)
					data = data.replace(np.inf, 0)
					# data=data.as_matrix()	
					header =pd.read_excel(root.filename).columns.tolist()
					i=1
					for each in header:
						dictionary[i]=each
						i=i+1
					btn.config(state="active")	
	


		else:
			label1.config(text="No option choosen",relief=RIDGE)


	except:
		label1.config(text="Incompatible format...Choose your file again",relief=RIDGE)	
		btn.config(state="disabled")	
#######################################################################################
def fillgap(btn):

	lstbox1.delete(0,END)
	lstbox1.insert(END, "No Labels choosen yet....")
	lstbox.delete(0,END)
	for key in sorted(dictionary):
		lstbox.insert(END, '{}: {}'.format(key, dictionary[key]))	
	btn.config(state="active")


	
	
# categorical_features : “all” 
#####################################################################################

def getlabel(btn):	
	global headerlabel
	attributeNumber=lstbox.curselection()
	headerlabel = StringVar()
	headerlabel=header[attributeNumber[0]]

	lstbox1.delete(0,END)
	for key in sorted(dictionary):
		lstbox1.insert(END, '{}: {}'.format(key, dictionary[key]))
	btn.config(state="active")	
#######################################################################################
def listboxselect(btn):
	global data
	global feature,output
	global dict1
	global featureList


	featureList = lstbox1.curselection()

	try:
		substData = data.copy(deep=True)

		dict1={}
		num=1
		for col in header:

			if substData[col].dtype.name == 'object':
				a = substData[col].unique()
				dict2={}
				for element in a:
					dict2[element]=num
					num=num+1
				dict1[col]=dict2
				num=1
		substData.replace(dict1, inplace=True)

	except Exception as e:
		print(e)


	out = substData[[headerlabel]].copy()


	df = pd.DataFrame()
	for i,each in enumerate(featureList):
		hl= header[featureList[i]]
		print(type(substData[[hl]]))

		df=pd.concat([df,substData[[hl]]],axis = 1)

	print(df)	
	
	feature=df.as_matrix()
	print(feature)
	print(type(feature))
	output=out.as_matrix()
	output=output.reshape(output.shape[0])
	btn.config(state="active")

#######################################################################################
def selectModel(b):
	global model
	X_train, X_test, y_train, y_test = train_test_split(feature, output, test_size=0.33)
	hp=IntVar()
	hp=hyppar.get()
	

	print(hp)
	print(type(hp))
	if(modeltype.get()==1):
		try:
			model = KNeighborsClassifier(n_neighbors=int(hp))
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			print(metrics.accuracy_score(y_test, pred))
			acc ="Predicted accuracy(Cross validation) is: "+str(metrics.accuracy_score(y_test, pred))
			label3.config(text=acc,relief=RIDGE)
			b.config(state="active")
			label4.config(text=" ")
		except:
			print("HERE")
			label3.config(text="Wrong Input Type",relief=RIDGE)
			print("Wrong input")	
			b.config(state="disabled")	
			label4.config(text="Choose correct value of Hyper-parameter",relief=RIDGE)	
			
	elif(modeltype.get()==2):	
		try:
			model = LogisticRegression(C=float(hp))
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			print(metrics.accuracy_score(y_test, pred))
			acc ="Predicted accuracy(Cross validation) is: "+str(metrics.accuracy_score(y_test, pred))
			label3.config(text=acc,relief=RIDGE)
			b.config(state="active")
			label4.config(text=" ")
		except:
			label3.config(text="Wrong Input Type",relief=RIDGE)
			print("Wrong input")
			b.config(state="disabled")
			label4.config(text="Choose correct value of Hyper-parameter",relief=RIDGE)
			
	elif(modeltype.get()==3):	
		try:	
			model = DecisionTreeClassifier(min_samples_split=int(hp))
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			print(metrics.accuracy_score(y_test, pred))
			acc ="Predicted accuracy(Cross validation) is: "+str(metrics.accuracy_score(y_test, pred))
			label3.config(text=acc,relief=RIDGE)
			b.config(state="active")
			label4.config(text=" ")
		except:
			label3.config(text="Wrong Input Type",relief=RIDGE)
			print("Wrong input")	
			b.config(state="disabled")
			label4.config(text="Choose correct value of Hyper-parameter",relief=RIDGE)

	elif(modeltype.get()==4):
		try:
			model = RandomForestClassifier(n_estimators=int(hp))
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			print(metrics.accuracy_score(y_test, pred))
			acc ="Predicted accuracy(Cross validation) is: "+str(metrics.accuracy_score(y_test, pred))
			label3.config(text=acc,relief=RIDGE)
			b.config(state="active")
			label4.config(text=" ")
		except:
			label3.config(text="Wrong Input Type",relief=RIDGE)
			print("Wrong input")
			b.config(state="disabled")
			label4.config(text="Choose correct value of Hyper-parameter",relief=RIDGE)	
	else:
		label3.config(text="No Option Selected",relief=RIDGE)

		
		
#######################################################################################
def getgt(bt):

	root.filename1 =  filedialog.askopenfilename()
	print(root.filename1)
	test_address=root.filename1
	root.filename2 =  filedialog.askopenfilename()
	print(root.filename2)
	gt_address=root.filename2
	print(type(model))

	for i,each in enumerate(featureList):
		print(featureList[i])


	data = pd.read_csv(test_address)
	data = data.replace(np.nan, 0)
	data = data.replace(np.inf, 0)
	data.replace(dict1, inplace=True) 

	try:
		test = pd.read_csv(root.filename2)
		test.replace(dict1, inplace=True)
		print(test)
		out = test[[headerlabel]].copy()
	except:
		print("Error in Ground truth File")
		label4.config(text="Error in Ground truth File",relief=RIDGE)
		return

	df = pd.DataFrame()
	for i,each in enumerate(featureList):
		hl= header[featureList[i]]

		df=pd.concat([df,data[[hl]]],axis = 1)		

	print(df)	
	feature=df.as_matrix()
	output = out.as_matrix()
	pred = model.predict(feature)
	print("Accuracy:",metrics.accuracy_score(output, pred))
	print(headerlabel)
	bt.config(state="disabled")
	label4.config(text="Accuracy:"+str(metrics.accuracy_score(output, pred)),relief=RIDGE)



#######################################################################################
address=""
data = pd.DataFrame()
header=[]
dictionary={}
feature=np.array([])
output=np.array([])

root = Tk()
root.title("Interactive ML GUI FOR CLASSIFICATION")
root.resizable(0,0)

frame = Frame(root)
frame.pack()

frame1 = Frame(root)
frame1.pack()

frame2 = Frame(root)
frame2.pack()

frame3 = Frame(root)
frame3.pack()

frame4 = Frame(root)
frame4.pack()

frame5 = Frame(root)
frame5.pack()

filetype=IntVar()
fillnull=IntVar()
modeltype=IntVar()
validation=IntVar()


#######################################################################################
#####################SHIVAM SECTION################################################

var = StringVar()
label = Label( frame, textvariable=var )
var.set("Select your File type:")
label.pack(pady=2)
R1 = Radiobutton(frame, text="CSV",variable = filetype, value=1)
R1.pack( anchor = W )
R2 = Radiobutton(frame,text="Excel File",variable = filetype,value=2)
R2.pack( anchor = W )

#######################################################################################
var = StringVar()
label = Label( frame, textvariable=var )
var.set("Enter your file here")
label.pack(pady=2)

button = Button(frame, text="Browse", fg="red",command=lambda : browseFile(button2) )
button.pack(pady=2)


#####################################################################
label1 = Label(frame)
label1.pack()

#####################################################################
var = StringVar()
label = Label( frame, textvariable=var )
var.set("Click here to get the list of labels in your choosen file:")
label.pack(pady=5)

button2 = Button(frame, text="Submit", fg="red",command=lambda : fillgap(button4),state="disabled")
button2.pack(pady=2)
label2 = Label(frame)
label2.pack()

#####################################################################
var5 = StringVar()
label5 = Label( frame1, textvariable=var5 )
var5.set("Choose the target label:")
label5.grid(row=0,column=0)

lstbox = Listbox(frame1)
lstbox.insert(END, "No file choosen yet....")
lstbox.grid(row=1,column=0)

button4 = Button(frame1, text="Select", fg="red",command=lambda : getlabel(button6),state="disabled")
button4.grid(row=2,column=0)

#####################################################################
var = StringVar()
label = Label( frame1, textvariable=var )
var.set("Feature Engineering:")
label.grid(row=0,column=1)

lstbox1 = Listbox(frame1,selectmode = "extended")
lstbox1.insert(END, "No Labels choosen yet....")
lstbox1.grid(row=1,column=1)

button6 = Button(frame1, text="Select", fg="red",command=lambda : listboxselect(button3) ,state="disabled")
button6.grid(row=2,column=1)

#####################################################################
#####################################################################
# Frame for grid layout
var = StringVar()
label = Label( frame2, textvariable=var )
var.set("Choose your Model:")
label.grid(row=0,column=0)
R1 = Radiobutton(frame2, text="KNeighborsClassifier  ", variable = modeltype, value=1)
R1.grid(row=1,column=0)
R2 = Radiobutton(frame2, text="LogisticRegression    ",variable = modeltype, value=2)
R2.grid(row=1,column=1)
R3 = Radiobutton(frame2, text="DecisionTreeClassifier", variable = modeltype, value=3)
R3.grid(row=2,column=0)
R4 = Radiobutton(frame2, text="RandomForestClassifier", variable = modeltype, value=4)
R4.grid(row=2,column=1)

label = Label( frame2,  text="Enter the hyper parameter here:")
label.grid(row=3,column=0)
hyppar = Entry(frame2)
hyppar.grid(row=3,column=1)


label3 = Label(frame3)
label3.grid(row=4,column=0,pady=2)


button3 = Button(frame4, text="Submit", fg="red",command=lambda : selectModel(button5),state="disabled")
button3.pack(pady=2)
#####################################################################

var = StringVar()
label = Label( frame5, textvariable=var )
var.set("Enter your Test file and Ground Truth file in respective order:")
label.pack(pady=2)

button5 = Button(frame5, text="Browse", fg="red",command=lambda : getgt(button5),state="disabled" )
button5.pack(pady=2)

label4 = Label(frame5)
label4.pack()



root.mainloop()



