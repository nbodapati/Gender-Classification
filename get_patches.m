function get_patches()
     male_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Male_Gender_cropped\'; 
     female_folder='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Female_Gender_cropped\'; 
   
     male_files=dir(male_folder);
     fprintf('Number of files in this folder: %d\n',length(male_files));
     
   for file_=1:length(male_files)
      if(male_files(file_).name(1)=='.')
         continue
      else
         %All are 250x250 
         %display(male_files(file_).name)
         img=imread(strcat(male_folder,male_files(file_).name));
         sobel_=edge(img,'sobel');
         
         features=vertcat(features,img(:).');
         labels=vertcat(labels,1);
         initials=vertcat(initials,male_files(file_).name(1));
         names=vertcat(names,string(male_files(file_).name));
         
       end
   end
        
   female_files=dir(female_folder);
   fprintf('Number of files in this folder: %d\n',length(female_files));
   
   for file_=1:length(female_files)
      if(female_files(file_).name(1)=='.')
         continue
      else
         %All are 250x250 
         img=imread(strcat(female_folder,female_files(file_).name));
         features=vertcat(features,img(:).');
         labels=vertcat(labels,0);
         initials=vertcat(initials,female_files(file_).name(1));
         names=vertcat(names,string(female_files(file_).name));
      end
   end
         
end