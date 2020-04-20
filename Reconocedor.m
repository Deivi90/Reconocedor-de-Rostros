%% Cargo Nueva Imagen
clear all

CantClass=10;     %Cantidad de personas
CantSamples=10;   %Cantidad de fotos por persona
Cant_TrainSamples=6;

%  [Rango,LDAfeatures,disc_vector,vectorClass,lenghtClass]=...
%             FeatureExtraction(16,CantClass,CantSamples,Cant_TrainSamples);
        
load('LDAfeatures.mat') % Carga los features de train
[FileName Path]=uigetfile('Imagenes\*.jpg','Abrir'); %Para buscar el archivo

%% Normalizacion de la imagen de test

image = imread(strcat(Path,FileName)); %Cargo todas las imagenes
image = rgb2gray(image);                        % Paso a gris

%Detector de ojos
    EyeDetect = vision.CascadeObjectDetector('RightEyeCART');
    eyedetected=step(EyeDetect,image); %[x y width height]
%Posicion de los ojos detectados
    Ojx=eyedetected(:,1)+eyedetected(:,3)/2;
    Ojy=eyedetected(:,2)+eyedetected(:,4)/2;   
% Busco el par de ojos. Se busca el par de ojos que esten alineados en el eje y :-)    
    MindistPair=1024;   %Inicializo con un valor alto
    for i = 1:size(eyedetected,1)-1
     
        [Min,Index]=min(abs(Ojy(i)-Ojy(i+1:end)));

        if MindistPair>Min
            MindistPair=Min;
            pair=[i,Index+i];
        end
    end
%Distancias porcentuales de la cara 

    c1=13/20; %Distancia entre los ojos 
    c2=10/20;   %Distancia desde ojo hasta bordes laterales de la imagen
    c3=6/20;    %Distancia desde ojo hasta borde superior de la imagen
    c4=20/20;   %Distancia desde ojo hasta borde inferior de la imagen

    Distojosx=abs(Ojx(pair(2))-Ojx(pair(1)));
    Distojosy=abs(Ojy(pair(2))-Ojy(pair(1)));

	%Recorto la imagen segun los valores definidos en C1 C2 C3 y C4
	
    Tam=Distojosx/c1;
    if (Ojx(pair(2)) < Ojx(pair(1))) %Me fijo que ojo esta mas a la derecha
	
        inx=Ojx(pair(2))-Tam*(1-c1)/2;	%Valor de la posicion x donde empieza la cara
        enx=Ojx(pair(1))+Tam*(1-c1)/2;	%Valor de la posicion x donde termina la cara
    else
        inx=Ojx(pair(1))-Tam*(1-c1)/2;
        enx=Ojx(pair(2))+Tam*(1-c1)/2;
    end
    iny=Ojy(pair(1))- Tam*c3;		%Valor de la posicion y donde empieza la cara
    eny=Ojy(pair(2))+Tam*c4;		%Valor de la posicion y donde termina la cara

    new=image(iny:eny,inx:enx);

% Guardo la imagen de la cara en la carpeta BasedeDatos

      new=imresize(new,[60 60]);	% La imagen de la cara sera de 60x60
%% Kmeans
%Convierto imagenes en vector.

vector = zeros(1,60*60);
k = 1;
for j = 1 : 60
    for i = 1 : 60
        vector(1,k) = new(j,i);
        k = k+1;
    end
end
% Uso las clases del kmeans para representar la imagen
featureVector=zeros(1,length(lenghtClass));
for i=1:length(vector)
    featureVector(1,vectorClass(i))=featureVector(1,vectorClass(i))+ vector(i);      
end
KfeatureVector=featureVector'./lenghtClass;

      

%% Clasificacion

trainFeatures=LDAfeatures;

% Paso los features por la matriz de LDA
testFeatures=disc_vect'*KfeatureVector;  
% Distancia Euclidea
testRepmat = repmat( testFeatures, 1,size(LDAfeatures,2));  %Matriz que contiene el valor a(t,:)en todas las filas    
dist = sqrt(sum(abs(testRepmat-LDAfeatures).^2, 1));
[minimum, index] = min(dist);        
Class_estim=ceil(index/Cant_TrainSamples);%Clase que se reconocio

if( minimum>50)% No funca tan bien como uno esperaria
    disp('El resultado no es muy acertado se estima que: ')
end


switch Class_estim
    case 1
        msgbox('La imagen es de Deivi')
    case 2
        msgbox('La imagen es de Pasto')
    case 3
        msgbox('La imagen es de Paiva')
    case 4
        msgbox('La imagen es de Tomas')
    case 5
        msgbox('La imagen es de Ari')
    case 6
        msgbox('La imagen es de Maso')
    case 7
        msgbox('La imagen es de Lea')
    case 8
        msgbox('La imagen es de Maru')     
    case 9
        msgbox('La imagen es de Nico')     
    case 10
        msgbox('La imagen es de Marcos')     
end
 
 
%