function Create_Dataset()
%This is the file that uses male_names.txt and female_names.txt to 
%read image files from folders and subfolders of lfw2 and 
%constructs two folders male_gender, female_gender with the images. 

    filepath=strcat('C:\Users\bodap\Documents\OntologyDataViz\GenderClassification','\lfw2\');
       
    output_male='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Male_Gender\';
    output_female='C:\Users\bodap\Documents\OntologyDataViz\GenderClassification\Female_Gender\';
    
    fid = fopen('male_names.txt');
    %Both are string arrays
    male_names_list=[];
    female_names_list=[];
    
    male_name_folders=[];
    female_name_folders=[];
    
    while(~feof(fid))
        male_names_list=[male_names_list string(fgetl(fid))];
    end
    fclose(fid);
    fprintf('Number of male names: %d \n',length(male_names_list));
    
    fid = fopen('female_names.txt');    
    while(~feof(fid))
        female_names_list=[female_names_list string(fgetl(fid))];
    end
    fclose(fid);
    fprintf('Number of female names: %d \n',length(female_names_list));
    
    %Loop over male names list:
    for name_idx=1: length(male_names_list)
        %convert a string into a char array
        text=char(male_names_list(name_idx));
        folder_name=[]; 
        for i =1:length(text)
            if (isletter(text(i)) || text(i)=='_' || text(i)=='-')
               folder_name=[folder_name text(i)]; 
            else
               break;
            end
        end
        
        if(length(male_name_folders)==0)
             male_name_folders=[male_name_folders string(folder_name(1:end-1))];
        end    
        
        if(~ismember(string(folder_name(1:end-1)),male_name_folders))
           %fprintf('NAme adding %s' ,string(folder_name(1:end-1))); 
           male_name_folders=[male_name_folders string(folder_name(1:end-1))];
        else
           %fprintf('NAme already present %s' ,string(folder_name(1:end-1)));
        end 
    end

    %Loop over female names list:
    for name_idx=1: length(female_names_list)
        %convert a string into a char array
        text=char(female_names_list(name_idx));
        folder_name=[]; 
        for i =1:length(text)
            if (isletter(text(i)) || text(i)=='_' || text(i)=='-')
               folder_name=[folder_name text(i)]; 
            else
               break;
            end
        end
        
        if(length(female_name_folders)==0)
             female_name_folders=[female_name_folders string(folder_name(1:end-1))];
        end    
        
        if(~ismember(string(folder_name(1:end-1)),female_name_folders))
           %fprintf('NAme adding %s\n' ,string(folder_name(1:end-1))); 
           female_name_folders=[female_name_folders string(folder_name(1:end-1))];
        else
           %fprintf('NAme already present %s\n' ,string(folder_name(1:end-1)));
        end 
    end
    
    %Loop over male folders and transfer all the contents to output_male
    %folder.
    fprintf('Number of male folders: %d\n',length(male_name_folders)) 
    fprintf('Number of female folders: %d\n',length(female_name_folders)) 
    
    for folder_num=1:length(male_name_folders)
        folder_name=char(male_name_folders(folder_num));
        full_folder_name=char(strcat(filepath,folder_name));
        files_list=dir(char(full_folder_name));
        fprintf('Number of files in this folder: %d\n',length(files_list));
        for file_ =1:length(files_list)
              %display(files_list(file_))
              if(files_list(file_).name(1)=='.')
                continue
              else   
                %display(strcat(files_list(file_).folder,'\',files_list(file_).name))
                %display(output_male)
                img=imread(strcat(files_list(file_).folder,'\',files_list(file_).name));
                imwrite(img,strcat(output_male,files_list(file_).name),'jpg');
                %cp2find(strcat(files_list(file_).folder,'\',files_list(file_).name),output_male);
              end
        
          end  
    end
    
    for folder_num=1:length(female_name_folders)
        folder_name=char(female_name_folders(folder_num));
        full_folder_name=char(strcat(filepath,folder_name));
        files_list=dir(char(full_folder_name));
        fprintf('Number of files in this folder: %d\n',length(files_list));
        for file_ =1:length(files_list)
             if(files_list(file_).name(1)=='.')
                continue
             else     
                %copyfile(strcat(files_list(file_).folder,'\',files_list(file_).name),output_female);
                img=imread(strcat(files_list(file_).folder,'\',files_list(file_).name));
                imwrite(img,strcat(output_female,files_list(file_).name),'jpg'); 
                %cp2find(strcat(files_list(file_).folder,'\',files_list(file_).name),output_female);
             end
        end
        
    end  
    
end 
