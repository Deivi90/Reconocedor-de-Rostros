%% Funcion Normalizacion de imagenes para la base de datos.
function CantFiles=NormalizacionRostro()

    %% Recorta toda la imagen quedando solamente la cara.

    files = dir('Imagenes\*.jpg');                 %Directorio de las Imagenes
    CantFiles=length(files);
    for k=1:length(files)

        image = imread(strcat('Imagenes\',files(k,1).name)); %Cargo todas las imagenes
        image = rgb2gray(image);                        % Paso a gris

    %Detector de ojos
        EyeDetect = vision.CascadeObjectDetector('RightEyeCART');
        eyedetected=step(EyeDetect,image); %[x y width height]
    %Posicion de los ojos detectados
        Ojx=eyedetected(:,1)+eyedetected(:,3)/2;
        Ojy=eyedetected(:,2)+eyedetected(:,4)/2;


    %Grafico ojos
    %     figure
    %     imshow(image); 
    %     hold on


    % Busco el par de ojos. Se busca el par de ojos que esten alineados en el eje y :-)    
        MindistPair=1024;   %Inicializo con un valor alto
        for i = 1:size(eyedetected,1)-1

            [Min,Index]=min(abs(Ojy(i)-Ojy(i+1:end)));

            if MindistPair>Min
                MindistPair=Min;
                pair=[i,Index+i];
            end
        end


    %     plot(Ojx,Ojy,'y+', 'MarkerSize', 5,'LineWidth',4);
    %     for j= 1:2
    %     rectangle('Position',BB(pair(j),:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
    %     end


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


    %     new=imresize(new,[60 60]);
    %     figure
    %     imshow(new)

    % Guardo la imagen de la cara en la carpeta BasedeDatos

          new=imresize(new,[60 60]);	% La imagen de la cara sera de 60x60
          imwrite(new,strcat('BasedeDatos\',files(k,1).name))
    end
end

