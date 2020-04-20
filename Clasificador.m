%% Clasificacion con la base de datos
clear all
CantClass=10;     %Cantidad de personas
CantSamples=10;   %Cantidad de fotos por persona
Cant_TrainSamples=6;   %Cantidad de fotos por persona par train
Cant_TestSamples=CantSamples-Cant_TrainSamples;

 [Rango,LDAfeatures,disc_vector,vectorClass,lenghtClass]=...
            FeatureExtraction(16,CantClass,CantSamples,Cant_TrainSamples);


LDAfeatures=LDAfeatures';

%% Separo FeaturesVector en vectores de Train y Test
        trainFeatures=LDAfeatures;

     Allfiles = dir('BasedeDatos\*.jpg');                                   %Directorio de las Imagenes
       %Agarro solo los files de TEST
        for k=1:CantClass  
                FilesClassK=Allfiles(...         
                CantSamples*(k-1)+1:...
                CantSamples*(k-1)+CantSamples,:);  
        Testfiles(Cant_TestSamples*(k-1)+1:...
                        Cant_TestSamples*(k-1)+Cant_TestSamples,:)=...
                           FilesClassK(Cant_TrainSamples+1:end,:);                                   
        end
        

        TestImages = zeros(60,60,length(Testfiles));
        for k=1:length(Testfiles)
            TestImages(:,:,k) = imread(strcat('BasedeDatos\',Testfiles(k,1).name));  %Cargo imagenes
        end  
        
        
        %Convierto imagenes en vector.
        vector = zeros(size(TestImages,3),60*60);
        k = 1;
        for j = 1 : 60
            for i = 1 : 60
                vector(:,k) = TestImages(j,i,:);
                k = k+1;
            end
        end
        % Uso las clases del kmeans para representar la imagen
        featureVector=zeros(CantClass*Cant_TestSamples,length(lenghtClass));
        for i=1:length(vector)
            featureVector(:,vectorClass(i))=featureVector(:,vectorClass(i))+ vector(:,i);      
        end
        testFeatures=featureVector'./(lenghtClass*ones(1,size(featureVector,1)));
       
        testFeatures=disc_vector'*testFeatures;  

        testFeatures=testFeatures';

%% Reconocimienot mediante distancia euclidea
correct=zeros(Cant_TestSamples*CantClass,1); %es igual a 1 si la deteccion es correcta
for i=1:Cant_TestSamples*CantClass
    
    testRepmat = repmat( testFeatures(i,:), size(trainFeatures,1), 1);  %Matriz que contiene el valor a(t,:)en todas las filas   
    dist = sqrt(sum(abs(testRepmat-trainFeatures).^2, 2));
    [minimum, index] = min(dist);        
   SampleClass=ceil(i/Cant_TestSamples);%Clase a la que pertenece la muestra de test
   Class_estim=ceil(index/Cant_TrainSamples);%Clase que se reconocio
    
    if (SampleClass==Class_estim)
        correct(i)=1;
    end
    
end

RecongnitionRate=sum(correct)/(Cant_TestSamples*CantClass);

    


    
    
    
    
    
    