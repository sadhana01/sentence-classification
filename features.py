from collections import Counter
import xml.sax
import pandas as pd
from Queue import *
frame=[]

df=pd.read_csv('allcorrectsenwlabels.txt',sep='\t',header=None)
df.columns=['Process','Sentence','Label']
#df['Frame']=[]
#print df
class FrameHandler( xml.sax.ContentHandler ):
   def __init__(self):
      self.CurrentData = ""
      self.frames=[]
      self.framelist={}
      self.num=0
   # Call when an element starts
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      #print "HEY"
      if tag == "sentence":
         #self.frames[]
        self.num= attributes["ID"]

      if self.CurrentData == "annotationSet":
        #print attributes["frameName"]
        #count=self.frames.count(attributes["frameName"])
        n= int(self.num)
        process=df.loc[n,'Process']
        self.frames.append(attributes["frameName"])
        if process not in self.framelist:
            self.framelist[process]=[]
        #count=self.framelist[process].count(attributes["frameName"])
        self.framelist[process].append(attributes["frameName"])

        #print self.frames
        """
        if count==0:
            self.framelist[process].append((count+1,attributes["frameName"]))
        else:
            self.framelist[process].remove((count,attributes["frameName"]))
            self.framelist[process].append((count+1,attributes["frameName"]))
        #print attributes["frameName"]
        """



   # Call when an elements ends
   def endElement(self, tag):

        self.CurrentData = tag
        if tag=="sentence":
            n= int(self.num)
            str=list(self.frames)
            #print "STR",list(str)
            #df.loc[n,'Frame']=str
            process=df.loc[n,'Process']
            #print process

            #self.framelist[process].append(self.frames)
            self.frames=[]
        self.CurrentData=""

# Call when a character is read
 #  def characters(self,tag, attributes):


#if ( __name__ == "__main__"):
def get_features():

   features={}
   # create an XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # override the default ContextHandler
   Handler = FrameHandler()
   parser.setContentHandler( Handler )

   parser.parse("allcorrectsen.txt.out")
   #Handler.framelist.sort(reverse=True)
   #print Handler.framelist
   #f.append(Handler.framelist[0])
   #f.append(Handler.framelist[1])
   #frame.append(f[0][1])
   #frame.append(f[1][1])
   for p in Handler.framelist:
        features[p]=[]
        count=Counter(Handler.framelist[p])
        #print "Process",p,"FRAMES",count.most_common(2)

        list=count.most_common(5)
        for i in range(len(list)):
            features[p].append(list[i][0])

   #print features
   #print features['absorption'][0][0]
   #df2=pd.DataFrame.from_dict(features)
   #df2.to_csv('correctfeatures3.csv', header=1,dtype=object)
   #return Handler.framelist
   return features

"""def get_framefeatures(inputfile):
    str=Processname+".txt.out"
    frame=frames.get_frame(str)
    train_features.append({"Process":Processname,"Frame1":frame[0],"Frame2":frame[1]}


"""
