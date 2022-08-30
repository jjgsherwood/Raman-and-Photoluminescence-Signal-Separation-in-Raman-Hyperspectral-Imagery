#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class HoribaThread(Window.RunningThread):
    def run(self):
        self.ExtDict = pickle.load(open(UserPref,"rb"))["Type_To_Ext"]
        self.PauseEvent.clear()
        wx.CallAfter(pub.sendMessage, "InputGaugeUpdate"+self.CommunicationID, msg=4,Text="Please check filename(s) and parameters")
        wx.CallAfter(pub.sendMessage,"GaugeFileName"+self.CommunicationID,FileName=False) #send the message to the dialog to ask the user for input
        self.PauseEvent.wait() #checks if the thread is pauzed
        if self.CheckClose():
            return
        self.NewFileName,self.ExcitingWavelength,self.TechniqueType,self.Pump,self.sensortype,self.BitDepth,self.date = self.child_conn.recv()
        wx.CallAfter(pub.sendMessage, "InputGaugeUpdate"+self.CommunicationID, msg=6,Text="Importing File") #update the dialog
       
        for i,FilePath in enumerate(self.FilePath):
            with open(FilePath) as f:
                lines = f.readlines()

            #check how it is split in colums and if it is , or . dec sperated
            if lines[1].count(",") == 0:
                ColumSplit = "\t"
                DecSplit = "."
            elif lines[1].count(".") == 0:
                ColumSplit = "\t"
                DecSplit = ","
            elif lines[1].count("\t") == 0:
                ColumSplit = ","
                DecSplit = "."
            else:
                wx.CallAfter(pub.sendMessage,"InputGaugeClose"+self.CommunicationID,ErrorMessage=["Unable to import the file because there are TABs ',' and '.' in the file. Please make sure that the colums are TAB or ',' serperated and that a '.' or ',' is used for decimal numbers and not both. Also the ',' can not be used for both.","IOError"])

            #Create Metadata dict
            self.MetaData = {"WaveNumberShiftInfo":map(float,lines[0].replace("\n","").replace(DecSplit,".").split(ColumSplit)[2:]),"Excitation Wavelength":float(self.ExcitingWavelength[i]),"Bit Depth":self.BitDepth[i],"sensor type":self.sensortype[i],"Technique Type":self.TechniqueType[i],"acquisition date":self.date[i]}          
            if self.MetaData["Technique Type"] == "Anti-Stokes Raman scattering":
                self.MetaData["Pump WL"] = self.Pump[i]
                self.MetaData["WavelengthInfo"] = map(lambda x: 1/((1/self.MetaData["Excitation Wavelength"])+(x*(10**-7))),self.MetaData["WaveNumberShiftInfo"])
            else:
                self.MetaData["WavelengthInfo"] = map(lambda x: 1/((1/self.MetaData["Excitation Wavelength"])-(x*(10**-7))),self.MetaData["WaveNumberShiftInfo"])
            if self.CheckClose():
                return

            #check the interleave of the ImportFile
            x0,y0 = [map(np.float32,lines[1].replace("\n","").replace(DecSplit,".").split(ColumSplit))[x] for x in [0,1]]
            x1,y1 = [map(np.float32,lines[2].replace("\n","").replace(DecSplit,".").split(ColumSplit))[x] for x in [0,1]]
            if x0 == x1:
                x_or_y = x0
                index = [0,1]
                self.MetaData["interleave"] = 'bip'
            elif y0 == y1:
                index = [1,0]
                x_or_y = y0
                self.MetaData["interleave"] = 'bipt'

            #checks the width and length of the image
            Block_DimLst = []
            for j,line in enumerate(lines[1:]):
                line = map(np.float32,line.replace("\n","").replace(DecSplit,".").split(ColumSplit))
                next_x_or_y,step = [line[n] for n in index]
                Block_DimLst.append(step)
                if next_x_or_y != x_or_y:
                    BlockLength = j
                    break
            if self.CheckClose():
                return

            #Determine the chunksize and the size of the image
            ImportPer = pickle.load(open(UserPref,"rb"))["LinesPerCore"]
            ChunksLambda = pickle.load(open(UserPref,"rb"))["ChunkSizeLambda"]
            Lines = len(lines[1:])/BlockLength if self.MetaData["interleave"] == 'bip' else BlockLength
            Width = BlockLength if self.MetaData["interleave"] == 'bip' else len(lines[1:])/BlockLength
            Bands = len(self.MetaData["WaveNumberShiftInfo"])
            if ImportPer > Lines:
                ImportPer = Lines
            self.MetaData["ChunkSize"] = (ImportPer,Width,ChunksLambda)
            self.MetaData["Image Type"] = "RHSI"
            self.MetaData["Shape"] = [Lines,Width,Bands]
            self.MetaData["Shape Original Data"] = [Lines,Width,Bands]
            self.MetaData["Special Resolution (micrometer)"] = 1

            #open the file,delete old file when overwrite and make the data set
            self.SaveName = Func.MakeFilePath(self.CurrentProject,self.NewFileName[i]+self.ExtDict["RHSI"])
            Func.OverwriteFileCheck([self.SaveName]) #if the file exist the user choose already that the file should be overwritten
            OutputFile = h5py.File(self.SaveName,'w')
            FileDataSet = OutputFile.create_dataset("Array",(Lines,Width,Bands),dtype=np.float32,chunks=self.MetaData["ChunkSize"],maxshape=(None,None,None))

            #reads and saves the ImportFile
            Slice = [0,0,slice(None)]
            Index_DimLst = []
            for line in lines[1:]:
                Array = np.array(line.replace("\n","").replace(DecSplit,".").split(ColumSplit),dtype=np.float32)
                Step,Array = Array[index[0]],Array[2:]
                Array[Array<0] = 1
                FileDataSet[tuple(Slice)] = Array
                Slice[index[1]] += 1
                if Slice[index[1]] == BlockLength:
                    Index_DimLst.append(Step)
                    Slice[index[1]] = 0
                    Slice[index[0]] += 1
                    progress = int(float(Slice[index[0]])/Lines * 95.)
                    wx.CallAfter(pub.sendMessage, "ProgressBarUpdate"+self.CommunicationID, msg=progress,FileNumber=i)
                    if self.CheckClose():
                        OutputFile.close()
                        return
            OutputFile.close()

            self.MetaData["Horizontal Dimension (px/mm)"] = 1000./np.diff(Index_DimLst).mean() if self.MetaData["interleave"] == 'bip' else 1000./np.diff(Block_DimLst[:-1]).mean()
            self.MetaData["Vertical Dimension (px/mm)"] = 1000./np.diff(Block_DimLst[:-1]).mean() if self.MetaData["interleave"] == 'bip' else 1000./np.diff(Index_DimLst).mean()
            self.MetaData["Aspect Ratio"] = self.MetaData["Horizontal Dimension (px/mm)"]/self.MetaData["Vertical Dimension (px/mm)"]

            TIPP.SaveMetaData(self.MetaData,self.CurrentProject,self.NewFileName[i]+self.ExtDict["RHSI"])
            TIPP.CreateLogEntry(self.CurrentProject,"Imported Image:",self.FileName[i],self.NewFileName[i])
            wx.CallAfter(pub.sendMessage, "ProgressBarUpdate"+self.CommunicationID, msg=100,FileNumber=i)
            wx.CallAfter(pub.sendMessage, "InputGaugeUpdate"+self.CommunicationID, msg=6+94/len(self.FileName)*(i+1))
        pub.sendMessage("InputGaugeClose"+self.CommunicationID) #closes the dialog

class HoribaGaugeDialog(Window.GaugeInputDialog):
    def MakePanels(self):
        self.Ext = pickle.load(open(UserPref,"rb"))["Type_To_Ext"]["RHSI"]
        self.OverWrite = [True]*len(self.FileName) #This list must contain only True if you want to continue. A Value is False if the file exist and you didn't aprove for overwrite.
        #checks how big the panel shall gets and desides if a scrollpanel is needed
        if len(self.FileName) > 15:
            size = (wx.ID_ANY,490)
        else:
            size = (wx.ID_ANY,len(self.FileName)*32+40)
        self.FileNamePanel = Cons.CreateGridFieldPanel(self.BasePanel,map(lambda x: [pt.splitext(x)[0],pt.splitext(x)[0],"785",["Stokes Raman scattering","Anti-Stokes Raman scattering"],"532","Horiba LabRAM HR Evolution Raman microspectrometer","16","1","1","2017","Overwrite",100],self.FileName),GridLayoutLst=["ST","TE","SP","CB","SP","TE","SP","SP","SP","SP","BU","GA"],Header=["Import Filename","New FileName","Excitation WL","Technique Type","Pump WL","sensor type","Bit Detph","Day","Month","Year","Overwrite?","Progress"],size=[0,0,80,0,0,0,0,45,45,55,70,100],SP_MaxVal=10**6,SP_MinVal=0,WindowSize=size) #The input field for filename
        self.FileNamePanel.SetupScrolling(False,True)
        self.FileNamePanel.Disables()
        pub.subscribe(self.FileNameInput, "GaugeFileName"+self.CommunicationID) #With this command the thread asks for the input of the filename
        pub.subscribe(self.UpdateProgressBar, "ProgressBarUpdate"+self.CommunicationID)

        for i in range(len(self.FileName)):
            self.FileNamePanel.GetItems(4)[i].Hide()
            self.Bind(wx.EVT_TEXT,lambda evt,FileNumber = i:self.CheckFileName(evt,FileNumber),self.FileNamePanel.GetItems(1)[i])
            self.Bind(wx.EVT_BUTTON,lambda evt,FileNumber = i:self.OnButton(evt,FileNumber),self.FileNamePanel.GetItems(10)[i])
            self.Bind(wx.EVT_COMBOBOX,lambda evt,FileNumber = i:self.OnChooseTechnique(evt,FileNumber),self.FileNamePanel.GetItems(3)[i])

    def MakeLayout(self):
        """
        Here is the layout of the dialog created
        """
        self.CommandoLst,self.PanelSize = [['V_N', 'C.A.B', [0, 0, 0], 1]] , [1,1] #The layout is made with all the input panels (otherwise there will be errors).
        self.Panellst = {"A":self.GaugePanel,"B":self.ButtonPanel,"C":self.FileNamePanel}
        self.MainPanel = Cons.PanelCustomLayout(self.BasePanel,self.Panellst,self.CommandoLst,self.PanelSize,Border=0)
        self.Fit()

    def OnButton(self,event,FileNumber):
        self.OverWrite[FileNumber] = True
        self.FileNamePanel.GetItems(10)[FileNumber].Disable()
        self.FileNamePanel.GetItems(10)[FileNumber].SetWindowStyleFlag(wx.NO_BORDER)
        self.FileNamePanel.GetItems(10)[FileNumber].SetBackgroundColour((0,255,0))
        if False in self.OverWrite:
            self.ButtonPanel.GetButtonList()[0].Disable()
        else:
            self.ButtonPanel.GetButtonList()[0].Enable()

    def CheckFileName(self,evt,FileNumber):
        """
        This looks if the file exist and changes the button name accordingly.
        """
        Path = glob.glob(pt.splitext(Func.MakeFilePath(self.CurrentProject,self.FileNamePanel.GetValues(1)[FileNumber]+self.Ext))[0]+".Tipp*")
        if Path != [] and self.OverWrite[FileNumber]:
            self.FileNamePanel.GetItems(10)[FileNumber].Enable()
            self.FileNamePanel.GetItems(10)[FileNumber].SetWindowStyleFlag(wx.SIMPLE_BORDER)
            self.FileNamePanel.GetItems(10)[FileNumber].SetBackgroundColour((230,230,230))
            self.FileNamePanel.GetItems(10)[FileNumber].SetForegroundColour((255,0,0))
            self.FileNamePanel.GetItems(10)[FileNumber].SetLabel("Overwrite")
            self.OverWrite[FileNumber] = False
        elif Path == []:
            self.OverWrite[FileNumber] = True
            self.FileNamePanel.GetItems(10)[FileNumber].Disable()
            self.FileNamePanel.GetItems(10)[FileNumber].SetWindowStyleFlag(wx.NO_BORDER)
            self.FileNamePanel.GetItems(10)[FileNumber].ClearBackground()
            self.FileNamePanel.GetItems(10)[FileNumber].SetBackgroundColour((0,255,0))
            self.FileNamePanel.GetItems(10)[FileNumber].SetLabel("New")
        if False in self.OverWrite:
            self.ButtonPanel.GetButtonList()[0].Disable()
        else:
            self.ButtonPanel.GetButtonList()[0].Enable()

    def OnChooseTechnique(self,evt,FileNumber):
        if self.FileNamePanel.GetValues(3)[FileNumber] == "Anti-Stokes Raman scattering":
            self.FileNamePanel.GetItems(4)[FileNumber].Show()
        else:
            self.FileNamePanel.GetItems(4)[FileNumber].Hide()

    def FileNameInput(self,FileName):
        """
        The thread asks for the filename so the part of the dialog that handels the filename gets enabled
        """
        self.OnNextCounter = "FileName" #sets the counter to Filename so OnNext knows which input is currently prossed
        self.ButtonPanel.GetButtonList()[0].Enable() #Enalbes the next button
        self.FileNamePanel.Enables() #Enables the FileName input field.
        for i in range(len(self.FileName)):
            self.CheckFileName(None,i)

    def OnNext(self,evt):
        """
        This def is called if the user clicks next and handles all the communication with the thread and disables the input fields
        """
        if self.OnNextCounter == "FileName": #checks which input is asks by the thread
            self.ButtonPanel.GetButtonList()[0].Disable() #Disables next
            self.FileNamePanel.Disables() #Disables the FileName input field
            self.parent_conn.send([self.FileNamePanel.GetValues(1),self.FileNamePanel.GetValues(2),self.FileNamePanel.GetValues(3),self.FileNamePanel.GetValues(4),self.FileNamePanel.GetValues(5),self.FileNamePanel.GetValues(6),map(lambda x,y,z:str(x)+"-"+str(y)+"-"+str(z),self.FileNamePanel.GetValues(7),self.FileNamePanel.GetValues(8),self.FileNamePanel.GetValues(9))]) #sent the filename to the thread
            self.DeleteSavedFile = True #after this point the file will be deleted when cancel is clicked
            self.PauseEvent.set() #unpauzes the thread

    def UpdateProgressBar(self,msg,FileNumber):
        """
        This updates the individual progressbars
        """
        self.FileNamePanel.SetValue(msg,FileNumber,10) #updates the gauge position

    def CloseGauge(self,evt=None,ErrorMessage=["",""]):
        """
        Handles the events if the dialog gets closed

        :param evt: This tells the user with which event this defintions is called. (default is None)
        :type evt: None or wx.event
        :param ErrorMessage: list with two strings. First string is the message the second the tile of the warningdialog.
        :type ErrorMessage: list
        """
        if evt != None: #If the dialog is closed by cancel evt != None
            self.CloseEvent.clear() #closes the thread
            self.PauseEvent.set() #activates the thread if it was pauzed otherwise the thread doesn't close
            if self.DeleteSavedFile: #checks if the saved file should be deleted
                ProcessLst = self.FileNamePanel.GetValues(10)
                delLst = filter(lambda x: ProcessLst[x] not in [0,100],range(len(self.FileName)))
                if len(delLst) != 0:
                    NewFilePath = Func.MakeFilePath(self.CurrentProject,self.FileNamePanel.GetValues(1)[delLst[0]]+self.Ext) #the filepath that will be delete
                    if pt.isfile(NewFilePath):
                        NewFileNameDialog = wx.MessageDialog(None,self.FileNamePanel.GetValues(1)[delLst[0]]+" already exist or is partly made, do you want to delete it? \n Partially created files are most likely corrupted.","Delete File?",style=wx.YES_NO|wx.ICON_QUESTION)
                        if NewFileNameDialog.ShowModal() == wx.ID_YES:
                            try:
                                remove(NewFilePath)
                            except:
                                pass
                            try:
                                remove(pt.splitext(NewFilePath)[0]+"_metadata.plk")
                            except:
                                pass        
        elif ErrorMessage != ["",""]: #If an error occurred a warning message can be made with ErrorMessage
            Func.WarningDialog(ErrorMessage[0],ErrorMessage[1])
        self.MakeModal(False)
        self.Destroy() 


