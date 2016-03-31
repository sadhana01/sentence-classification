from collections import Counter
import xml.sax
import pandas as pd
from Queue import *
frame=[]

df=pd.read_csv('senwlabels.txt',sep='\t',header=None)
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
        if n not in self.framelist:
            self.framelist[n]=[]

        self.framelist[n].append(attributes["frameName"])




   # Call when an elements ends
   def endElement(self, tag):

        self.CurrentData = tag
        if tag=="sentence":
            n= int(self.num)
            str=list(self.frames)

            process=df.loc[n,'Process']
            #print process


            self.frames=[]
        self.CurrentData=""

# Call when a character is read
 #  def characters(self,tag, attributes):


#if ( __name__ == "__main__"):
def get_frames():

   features={}
   # create an XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # override the default ContextHandler
   Handler = FrameHandler()
   parser.setContentHandler( Handler )

   parser.parse("sen.txt.out")

   return Handler.framelist



