function CropScaleImages()
   male_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Male_Gender\';     
   female_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Female_Gender\';
   
   cropped_male_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Male_Gender_cropped\'; 
   cropped_female_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Female_Gender_cropped\'; 
   
   features=[];
   labels=[];
   initials=[];
   names=[];
   
   male_files=dir(male_folder);
   fprintf('Number of files in this folder: %d\n',length(male_files));
   for file_=1:length(male_files)
      if(male_files(file_).name(1)=='.')
         continue
      else
         %All are 250x250 
         %display(male_files(file_).name)
         img=imread(strcat(male_folder,male_files(file_).name));
         %imtool(img)
         img2=imcrop(img,[75,60,(175-75),(200-60)]);
         img2=imresize(img2,[60,60]);
         features= vertcat(features,img2);
         labels=vertcat(labels,1);
         initials=vertcat(initials,male_files(file_).name(1));
         names=vertcat(names,string(male_files(file_).name));
         imwrite(img2,strcat(cropped_male_folder,male_files(file_).name),'jpg')
       end
   end
        
     female_files=dir(female_folder);
   fprintf('Number of files in this folder: %d\n',length(female_files));
   for file_=1:length(female_files)
      if(female_files(file_).name(1)=='.')
         continue
      else
         %All are 250x250 
         %display(male_files(file_).name)
         img=imread(strcat(female_folder,female_files(file_).name));
         %imtool(img)
         img2=imcrop(img,[75,60,(175-75),(200-60)]);
         img2=imresize(img2,[60,60]);
         features= vertcat(features,img2);
         labels=vertcat(labels,0);
         initials=vertcat(initials,female_files(file_).name(1));
         names=vertcat(names,string(male_files(file_).name));
         
         imwrite(img2,strcat(cropped_female_folder,female_files(file_).name),'jpg')
       end
   end
        
   f_l_i=horzcat(features,labels,initials,names);
   csvwrite('Processed_LFW_dataset.csv',f_l_i);
   
end