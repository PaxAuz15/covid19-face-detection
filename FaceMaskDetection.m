classdef FaceMaskDetection < vision.labeler.AutomationAlgorithm
    
    properties(Constant)
        
        Name = 'Face Mask Detection';        
        Description = 'Este es un algoritmo automático de etiquetado de mascarilla. .';
 
        UserDirections = {...
            ['Los algoritmos de automatización son una forma de automatizar ' ...
            'las tareas de etiquetado manual . Este algoritmo de automatización ' ...
            'es una plantilla para crear algoritmos de automatización definidos ' ...
            ' por el usuario. A continuación se muestran los pasos típicos ' ...
            'involucrados en la ejecución de un algoritmo de automatización.'], ...
            ['Ejecutar: presione RUN para ejecutar el algoritmo de automatización.'],...
            ['Revisar y modificar: revise las etiquetas automatizadas ' ...
            'durante el intervalo mediante los controles de reproducción. ' ...
            'Modifique / elimine / agregue ROI que no se automatizaron ' ...
            'satisfactoriamente en esta etapa. Si los resultados son ' ...
            'satisfactorios, haga clic en Aceptar para aceptar las ' ...
            'etiquetas automáticas'], ...
            ['Aceptar / Cancelar: si los resultados de la automatización ' ...
            'son satisfactorios,haga clic en Aceptar para aceptar todas las ' ...
            'etiquetas automáticas y volver al etiquetado manual. Si los ' ...
            'resultados de la automatización no son satisfactorios, haga clic ' ...
            'en Cancelar para volver al etiquetado manual sin guardar las '...
            'etiquetas automáticas.']};
    end
    
    properties
        
        AllCategories = {'background'};
        FireName
        count
        
        
    end
    
    methods
        function isValid = checkLabelDefinition(algObj, labelDef)
            
            disp(['Executing checkLabelDefinition on label definition "' labelDef.Name '"'])
            
            if (strcmpi(labelDef.Name, 'Mask') && labelDef.Type == labelType.Rectangle)
                isValid = true;
                algObj.FireName = labelDef.Name;
                algObj.AllCategories{end+1} = labelDef.Name;
            end           
            
            
        end
        
        function isReady = checkSetup(algObj)
            
            isReady = ~isempty(algObj);          

            
            
        end
        
        function settingsDialog(algObj)
            
            disp('Ejecutando settingsDialog')

        end
    end
    
    methods

        function initialize(algObj, I)
            
            disp('Ejecutando initialize en el primer cuadro de imagen')

        end

        function autoLabels = run(algObj, I)
            
            disp('Ejecutando ejecutar en marco de imagen')
            
            [labelCord, label] = MaskLabel(I, algObj);                
            autoLabels.Name = char(label);
            autoLabels.Type = labelType('Rectangle');
            autoLabels.Position = labelCord;               
            algObj.count = algObj.count+1;
            
        end

        function terminate(algObj)
            
            disp('Ejecutando terminar')
            
        end
    end
end
% Copyright 2020 The MathWorks, Inc.