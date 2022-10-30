x = [-1.2;1];
A = 100;


%gradient_descent(x, A);

x = [-1.2;1];
A = 100;
%newton_method(x, A);

x = [-1.2;1];
A = 1;
gradient_descent(x, A);

x = [-1.2;1];
A = 1;
newton_method(x, A);

figure
%plot of cost function
title('Plot of Cost Function');
[X,Y]=meshgrid(-2:0.1:2);
Z=100*(Y-X.^2).^2+(ones(size(X))-X).^2;
surf(X,Y,Z)

function gradient_descent(x, A)

    figure;
    hold on;
    xlim([-10 90]);
    xlabel('Number of Iterations');
    ylabel(['Value of the norm of the gradient']);
    if(A == 100) 
        title('Value of Norm of Gradient vs. Number of Iterations, A = 100, Gradient Descent'); 
        ylim([0 55]);
    end
    if(A == 1) 
        title('Value of Norm of Gradient vs. Number of Iterations, A = 1, Gradient Descent'); 
        ylim([0 5]);
    end
    
    storage = zeros(2, 6000); %storage for path plot
    c = 0.01;
    rho = .5;
    f=func(x, A);
    g=grad(x, A);
    k = 0; % k = # iterations
    funcEval=1;	% funcEval = # function eval.	
    
    
    % Begin method
    while ( norm(g) > 1e-3 )
        if(norm(g) > 1e-2) plot(k, norm(g), '.'); end
        pk = -g; % steepest descent direction
        a = 1;
        newf = func(x + a*pk, A);
        funcEval = funcEval+1;
        while (newf > f + c*a*g'*pk)
            a = a*rho;
            newf = func(x + a*pk, A);
            funcEval = funcEval+1;
        end
    
        x = x + a*pk; % gradient descent
        storage(1, k+1) = x(1);
        storage(2, k+1) = x(2);
    
        f=newf;
        g=grad(x, A);
        k = k + 1;
    end
    celltable = cell(1,3);
    celltable{1,1} = transpose(x);
    celltable{1,2} = k;
    celltable{1,3} = funcEval;
    
    if(A == 100)
        T = cell2table(celltable,...
        "VariableNames",["Estimate using Gradient Descent, A = 100" "Number of Iterations" "Number of Function Calls"]);
    end
    if(A == 1)
        T = cell2table(celltable,...
        "VariableNames",["Estimate using Gradient Descent, A = 1" "Number of Iterations" "Number of Function Calls"]);
    end
    disp(' ');
    disp(T);

    figure
    hold on
    
    
    if(A == 100) 
        title('Path of Optimization, A = 100, Gradient Descent'); 
        xlim([0.67, 1.25]);
        ylim([0, 1.5]);
    end
    if(A == 1) 
        title('Path of Optimization, A = 1, Gradient Descent'); 
        xlim([0, 1.5]);
        ylim([0.65, 1.25]);
    end 


    contour = @(x,y) 100*(y-x^2)^2 + (1-x)^2;
    fcontour(contour, 'MeshDensity', 200)
    
    plot(storage(1, 1:k-1), storage(2, 1:k-1), 'kx-');
    plot(storage(1, k), storage(2, k), 'ro', 'MarkerFaceColor', 'r');

end



function newton_method(x, A)

    figure;
    hold on;
    xlim([0 90]);
    xlabel('Number of Iterations');
    ylabel(['Value of the norm of the gradient']);
    if(A == 100) 
        title('Value of Norm of Gradient vs. Number of Iterations, A = 100, Newton Method'); 
        ylim([0 90]);
    end
    if(A == 1) 
        title('Value of Norm of Gradient vs. Number of Iterations, A = 1, Newton Method'); 
        ylim([0 10]);
    end
    
    storage = zeros(2, 200); %storage for path plot
    c = 0.01;
    rho = .5;
    f=func(x, A);
    g=grad(x, A);
    h = hessian(x, A);
    k = 0; % k = # iterations
    funcEval=1;	% funcEval = # function eval.	
    
    
    % Begin method
    while ( norm(g) > 1e-3 )
        if(norm(g) > 1e-2) plot(k, norm(g), '.'); end
        pk = -1*(h \ g); % steepest descent direction
        a = 0.1;
        newf = func(x + a*pk, A);
        funcEval = funcEval+1;
        while (newf > f + c*a*g'*pk)
            a = a*rho;
            newf = func(x + a*pk, A);
            funcEval = funcEval+1;
        end
    
        x = x - a*(h \ g); % newton method
        storage(1, k+1) = x(1);
        storage(2, k+1) = x(2);
    
        h = hessian(x, A);
        f=newf;
        g=grad(x, A);
        k = k + 1;
    end

    
    celltable = cell(1,3);
    celltable{1,1} = transpose(x);
    celltable{1,2} = k;
    celltable{1,3} = funcEval;
    
    if(A == 100)
        T = cell2table(celltable,...
        "VariableNames",["Estimate using Newton Method, A = 100" "Number of Iterations" "Number of Function Calls"]);
    end
    if(A == 1)
        T = cell2table(celltable,...
        "VariableNames",["Estimate using Newton Method, A = 1" "Number of Iterations" "Number of Function Calls"]);
    end
    disp(' ');
    disp(T);

    
    figure
    hold on
    
    
    if(A == 100) 
        title('Path of Optimization, A = 100, Newton Method');
        xlim([-1.5, 1.5]);
        ylim([-0.4, 1.2]);
    end
    if(A == 1) 
        title('Path of Optimization, A = 1, Newton Method'); 
        xlim([-1.5, 1.5]);
        ylim([-0.3, 1.25]);
    end


    contour = @(x,y) 100*(y-x^2)^2 + (1-x)^2;
    fcontour(contour, 'MeshDensity', 200)
    
    plot(storage(1, 1:k-1), storage(2, 1:k-1), 'kx-');
    plot(storage(1, k), storage(2, k), 'ro', 'MarkerFaceColor', 'r');

end


function y = func(x, A)
    y = A*(x(1)^2 - x(2))^2 + (x(1)-1)^2;
end

function y = grad(x, A)
    y(1) = A*(2*(x(1)^2-x(2))*2*x(1)) + 2*(x(1)-1);
    y(2) = A*(-2*(x(1)^2-x(2)));
    y = y';
end

function y = hessian(x, A)
    y(1,1) = (12*A) * x(1)^2 - (4*A) * x(2) + 2;
    y(1,2) = -(4*A) * x(1);
    y(2,1) = -(4*A) * x(1);
    y(2,2) = (2*A);
end