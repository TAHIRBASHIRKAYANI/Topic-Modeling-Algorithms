# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:38:17 2019

@author: TAHIR BASHIR KAYANI
"""

from owlready2 import *
import os
i = 1;
j=1
#r'E:\MSSE\newtestdata\ow'

path = r'G:\Synopsis Tahir\LDATOPICMODELLING\ontology data set'
for path, dirs, files in os.walk(path):
    #I know none of my subdirectories will have their own subfolders 
    if len(dirs) == 0:
        #print("Subdirectory name:", os.path.basename(path))
        print(i)
#        print('************')
        i=i+1
#        print(path)
#        break
       # print("Files in subdirectory:", ', '.join(files))
        for filename in os.listdir(path):
            if not filename.endswith('.owl'): continue
            fullname = os.path.join(path, filename)
           # print(fullname)
            #print(i)
            onto = get_ontology(fullname).load() #https://pythonhosted.org/Owlready2/onto.html
           # print(onto)
            
#            print(j)
#            print('##########'+ path)
        
            j=j+1
            onto = onto.load()
            classes = list(onto.classes())
            properties = list(onto.properties())
            annotation_properties = list(onto.annotation_properties())
            
            data_properties = list(onto.data_properties())
            
            object_properties = list(onto.object_properties())
            all_triples = onto.get_triples(None, None, None) #https://en.wikipedia.org/wiki/Semantic_triple
            metadata = onto.metadata
            
            print(metadata)
         
            # ontology_comment=list(onto.object_lab())
            #print(list(all_triples))
#            print("pat to save tripple is " + path)
#            onto.save(r'G:\Synopsis Tahir\LDATOPICMODELLING\ontology data set\statistical'+ filename, format = "ntriples")
#            print("saved files")
            #onto.save(r'G:\Synopsis Tahir\LDATOPICMODELLING\ontology data set'+filename, format = "ntriples") 
#            metadata = onto.metadata.comment
#            metadata_label=onto.metadata.label
##            print(metadata)
##            print("\n Clases \n" )
##            print(classes)
##            print("\n Properties \n" )
##            print(properties)
##            print("\n Annotation_properties \n" )
##            print(annotation_properties)
#            print("\n metadat comment\n" )
#            print(metadata)
#            print("\n metadata label \n" )
#            print(metadata_label)
##            print(triples)
#            i= i + 1;
##            #f= open(i+".txt","w+")
#            with open(path +filename + 'terms.txt', 'w') as f:
#                for item in metadata_label,metadata,classes,properties,annotation_properties:
#                    f.write("%s\n" % item)
#            with open(path +filename + 'metadata.txt', 'w') as f:
#               for item in metadata:
#                   f.write("%s\n" % item)
#            with open(path +filename + 'triples.txt', 'w') as f:
#               #for item in triples:
#                   f.write("%s\n" % list(all_triples))