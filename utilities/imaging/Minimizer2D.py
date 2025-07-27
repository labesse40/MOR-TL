import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np
from scipy import optimize


class Minimizer2D:
    """
    Class containing methods for the minimization of scalar function of one or more variables\
    This class is compatible with scipy.optimize methods

    See scipy.optimize documentation for an exhaustive list of parameters:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """
    def __init__(self, func, method="steepestDescent", jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            func : callable
                Method returning the cost function value
                When jac is set to None, func can also be a method returning both the functionnal value and the gradient value (J, gradJ). Note that this option does not work with 'steepestDescent' due to
                optimizations call in the method
            method : str
                Optimization method to be used \
                Default is 'steepest descent'
            jac : callable, optional
                Method returning the gradient of the functional
            hess : callable, optional
                Method returning the hessian of the functionnal
            hessp : callable, optional
                Hessian of objective function times an arbitrary vector p. Only for Newton-CG, trust-ncg, trust-krylov, trust-constr. Only one of hessp or hess needs to be given. If hess is provided,
                then hessp will be ignored. hessp must compute the Hessian times an arbitrary vector.
            bounds : list, optional
                Bounds on variable. Sequence of (min, max) pairs for each element in x.
            constraints : {Constraint, dict} or list of {Constraint, dict}
                Constraints definition. Only for COBYLA, SLSQP and trust-constr.
                Constraints for `trust-constr` are defined as a single object or a list of objects specifying constraints to the optimization problem.
            tol : float, optional
                Tolerance for termination. When tol is specified, the selected minimization algorithm sets some relevant solver-specific tolerance(s) equal to tol.
                For detailed control, use solver-specific options.
            callback : callable, optional
                Called after each iteration. The signature is: 'callback(xk)', where xk is the current parameter vector. For `trust-constr` method, see the Scipy.optimize documentation
            options : dict, optional
                A dictionary of methods options. See the scipy.optimize documentation for all the different options for the different methods. Or see below for 'steepestDescent' options
        """
        self.func = func
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options


    def optimize(self, x, args):
        """
        Call the optimization method

        Parameters:
        -----------
            x : array
                Current value of the variable to optimize
            args: tuple
                List of func and jac callables arguments
        """
        if self.method == "steepestDescent":
            result = SteepestDescentOptimizer( self.func, self.jac, self.hess, self.hessp, self.bounds, self.constraints, self.tol, self.callback, self.options ).optimize( x, args )

        else:
            result = optimize.minimize(self.func,
                                       x,
                                       args=args,
                                       method=self.method,
                                       jac=self.jac,
                                       hess=self.hess,
                                       hessp=self.hessp,
                                       bounds=self.bounds,
                                       constraints=self.constraints,
                                       tol=self.tol,
                                       callback=self.callback,
                                       options=self.options)

            return result


class SteepestDescentOptimizer(Minimizer2D):
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                p1 : float, optional
                    Coefficient for line search step re-evaluation if the step is to low
                    Default is 1.5
                p2 : float, optional
                    Coefficient for line search step re-evaluation if the step is to large
                    Default is 0.5
                minls : int, optional
                    Minimum number of evaluations in line search
                    Default is 3
                maxls : int, optional
                    Maximum number of evaluations in line search
                    Default is the value of `minls` parameter
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
                nbCfcheck : int, optional
                    Number of previous cost function evaluation to check if the functionnal decreases enough
                    Default is 5
                wolfeConditions: bool, optional
                    Activate Wolfe conditions
                    Default is False
                k1 : float, optional
                    Coefficient for lower bound for wolfe conditions
                    Default is 0.2
                k2 : float, optional
                    Coefficient for upper bound for wolfe conditions
                    Default is 0.8
        """
        super().__init__(func, "steepestDescent", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}

        self.alpha = options.get("alpha", 1e-8)
        self.p1 = options.get("p1", 1.5)
        self.p2 = options.get("p2", 0.5)
        self.minls = options.get("minls", 3)
        self.maxls = options.get("maxls", self.minls)
        self.miniter = options.get("miniter", 1)
        self.maxiter = options.get("maxiter", 10)
        self.nbCFcheck = options.get("nbCFcheck", 5)
        self.wolfeConditions = options.get("wolfeConditions", False)
        self.k1 = options.get("k1", 0.2)
        self.k2 = options.get("k2", 0.8)

        if self.maxls < self.minls:
            self.maxls = self.minls

        if self.maxiter < self.miniter:
            self.maxiter = self.miniter

    def optimize(self, x, args):
        """
        Optimize with steepest descent method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0
        success = 0
        checkJ = np.zeros(self.nbCFcheck)
        model = args[6] # model parametrization

        print(f"\n\nStarting steepest descent optimization\n", flush=True)
        while iter < self.maxiter and not success:

            lineSearchIteration = 0
            alpha = self.alpha

            if iter == 0:
                _, d, J = self.jac( x, *args, computeFullCF=True )
                checkJ[0] = J

            else:
                _, d = self.jac( x, *args )
                J = Jup

            if max(abs(d)) > 0:
                d = - d / max(abs(d))
            else:
                d = np.zeros_like(d)  # Set d to zero if all elements are zero

            with open("fullCostFunction.txt", "a") as f:
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch iteration {lineSearchIteration} - Full cost function : {J}\n", flush=True)

            while True:
                print(f"Line search iteration {lineSearchIteration} - alpha = {alpha}", flush=True)
                xup = x + alpha * d[:]
                # Constraining model to bounds
                xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                if model == "1/c2":
                    print(f"Maximum velocity: {np.sqrt(1/np.max(xup))}", flush=True)
                    print(f"Minimum velocity: {np.sqrt(1/np.min(xup))}", flush=True)
                elif model == "1/c":
                    print(f"Maximum velocity: {1/np.max(xup)}", flush=True)
                    print(f"Minimum velocity: {1/np.min(xup)}", flush=True)
                else:
                    print(f"Maximum velocity: {np.max(xup)}", flush=True)
                    print(f"Minimum velocity: {np.min(xup)}", flush=True)


                # Check if model perturbation is significant (avoid local minimum).
                # Using 1e7. This controls the accuracy, 1e12 provides low accuracy,
                # 1e7 moderate, and 10 extremely high (scipy Notes).
                epsmch = np.finfo(float).eps
                factr = 1e7*epsmch
                # This condition will need to be improved cause it only applies for the acoustic case
                if (
                    (model == "c" and (((x - xup) / max(np.max(abs(x)), np.max(abs(xup)), 1)) <= factr).all())
                    or (model == "1/c" and (((1 / x - 1 / xup) / max(np.max(abs(1 / x)), np.max(abs(1 / xup)), 1)) <= factr).all())
                    or (model == "1/c2" and (((np.sqrt(1 / x) - np.sqrt(1 / xup)) / max(np.max(abs(np.sqrt(1 / x))), np.max(abs(np.sqrt(1 / xup))), 1)) <= factr).all())
                ):
                    print( "*"*50, flush=True )
                    print("WARNING: model perturbation is too small, stopping.", flush=True)

                    success = 1

                    break

                Jup = self.func( xup, *args )

                if lineSearchIteration < self.minls and lineSearchIteration < self.maxls:
                    # Adapt alpha depending on Jup and J values
                    if Jup >= J:
                        # Lower alpha
                        alpha *= self.p2

                    else:
                        # Increase alpha
                        alpha *= self.p1

                    lineSearchIteration += 1

                # Min and max number of linesearch iter reached (only when minls == maxls)
                elif lineSearchIteration == self.minls and lineSearchIteration == self.maxls:
                    lineSearchIteration += 1

                    print( "*"*50, flush=True )
                    print( f"FWI iteration {iter} - Max number of linesearch iterations reached \n Last {self.nbCFcheck} full cost functions computed : {checkJ}\n", flush=True )

                    if not all( checkJ ):
                        if Jup >= J:
                            alpha *= 0.1 * self.p2**self.maxls
                            xup = x + alpha * d

                            # Constraining model to bounds
                            xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                        x = xup

                        if self.callback is not None:
                            self.callback( x )

                        iter += 1
                        checkJ[iter % self.nbCFcheck] = Jup

                        break

                    # If checkJ is completely filled
                    else:
                        mean = np.mean( checkJ )
                        # Change the frequency regardless of the value of Jup
                        # If Jup has increased, keep last model, otherwise update model
                        if Jup >= 0.95 * mean:
                            success = 1

                            break

                        else:
                            x = xup

                            if self.callback is not None:
                                self.callback( x )

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                # Minimum number of linesearch iterations reached
                elif lineSearchIteration >= self.minls and lineSearchIteration < self.maxls:
                    lineSearchIteration += 1

                    if not all( checkJ ):
                        if Jup < J:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            alpha *= self.p2

                    else:
                        mean = np.mean( checkJ )

                        if Jup >= 0.95 * mean and Jup <= 1.05 * mean:
                            success = 1

                            break

                        elif Jup < 0.95 * mean:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            alpha *= self.p2

                elif lineSearchIteration >= self.maxls:
                    alpha *= 0.1 * self.p2**self.maxls

                    xup = x + alpha * d

                    # Constraining to bounds
                    xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                    x = xup

                    if self.callback is not None:
                        self.callback( x )

                    iter += 1
                    checkJ[iter % self.nbCFcheck] = Jup

                    break

                # Extra criteria to avoid dwelling on a local minimum
                if np.abs(Jup - J) < 1e-8:
                    print( "*"*50, flush=True )
                    print("Cost function does not change, stopping optimization.", flush=True)

                    success = 1

                    break

            # Ensure a minimum amount of iterations
            if iter < self.miniter:
                if success:
                    success = 0
                    iter += 1

        return x




    def WolfeConditions(self, J, Jup, d, gradJ):
        """
        Check Wolfe conditions

        Parameters
        -----------
            J : float
                Value of the cost function
            Jup : float
                Value of the cost function for perturbed model
            d : array
                Descent direction

        Returns
        --------
            bool :
                0 if conditions not verified, 1 otherwise
        """
        succesArmijo = self.armijoCondition(J, Jup, d, gradJ)

        if succesArmijo:
            succesGoldstein = self.goldsteinCondition(J, Jup, d, gradJ)

            if succesGoldstein:
                return 1
            else:
                self.alpha *= self.p1
                return 0
        else:
            self.alpha *= self.p2
            return 0


    def armijoCondition(self, J, Jm, d, gradJ):
        """
        Check Armijo condition

        Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm <= J + self.k1 * self.alpha * np.dot(d, gradJ))

        return success


    def goldsteinCondition(self, J, Jm, d, gradJ):
        """
        Check Goldstein condition

       Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm >= J + self.k2 * self.alpha * np.dot(d, gradJ))

        return success





class SteepestDescentOptimizerROM(Minimizer2D):
    def __init__(self, func, jac=False, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """
        Parameters
        -----------
            options:
                alpha: float, optional
                    Starting line search step
                    Default is 1e-8
                p1 : float, optional
                    Coefficient for line search step re-evaluation if the step is to low
                    Default is 1.5
                p2 : float, optional
                    Coefficient for line search step re-evaluation if the step is to large
                    Default is 0.5
                minls : int, optional
                    Minimum number of evaluations in line search
                    Default is 3
                maxls : int, optional
                    Maximum number of evaluations in line search
                    Default is the value of `minls` parameter
                maxiter : int, optional
                    Maximum number of iterations
                    Default is 10
                nbCfcheck : int, optional
                    Number of previous cost function evaluation to check if the functionnal decreases enough
                    Default is 5
                wolfeConditions: bool, optional
                    Activate Wolfe conditions
                    Default is False
                k1 : float, optional
                    Coefficient for lower bound for wolfe conditions
                    Default is 0.2
                k2 : float, optional
                    Coefficient for upper bound for wolfe conditions
                    Default is 0.8
        """
        super().__init__(func, "steepestDescent", jac, hess, hessp, bounds, constraints, tol, callback, options)

        if options is None:
            options = {}

        self.alpha = options.get("alpha", 1e-8)
        self.p1 = options.get("p1", 1.5)
        self.p2 = options.get("p2", 0.5)
        self.minls = options.get("minls", 3)
        self.maxls = options.get("maxls", self.minls)
        self.miniter = options.get("miniter", 1)
        self.maxiter = options.get("maxiter", 10)
        self.nbCFcheck = options.get("nbCFcheck", 5)
        self.wolfeConditions = options.get("wolfeConditions", False)
        self.k1 = options.get("k1", 0.2)
        self.k2 = options.get("k2", 0.8)

        if self.maxls < self.minls:
            self.maxls = self.minls

        if self.maxiter < self.miniter:
            self.maxiter = self.miniter



    def optimize(self, x, args):
        """
        Optimize with steepest descent method

        Parameters
        -----------
            x : array
                Current value of the variable to optimize
            args: tuple
                List of func and jac callables arguments
        """
        iter = 0
        Jup = 0
        success = 0
        checkJ = np.zeros(self.nbCFcheck)
        model = args[6] # model parametrization

        print(f"\n\nStarting steepest descent optimization\n", flush=True)
        while iter < self.maxiter and not success:

            lineSearchIteration = 0
            alpha = self.alpha

            if iter == 0:
                _, d, J = self.jac( x, *args, computeFullCF=True )
                checkJ[0] = J

            else:
                _, d = self.jac( x, *args )
                J = Jup

            if max(abs(d)) > 0:
                d = - d / max(abs(d))
            else:
                d = np.zeros_like(d)  # Set d to zero if all elements are zero

            with open("fullCostFunction.txt", "a") as f:
                f.write(f"{iter} \t {str(J)}\n")

            print(f"\nFWI iter {iter} - linesearch iteration {lineSearchIteration} - Full cost function : {J}\n", flush=True)

            while True:
                print(f"Line search iteration {lineSearchIteration} - alpha = {alpha}", flush=True)
                xup = x + alpha * d[:]
                # Constraining model to bounds
                imin = np.where( xup < self.bounds[0][0] )[0]
                imax = np.where( xup > self.bounds[0][1] )[0]
                xup[imin] = self.bounds[0][0]
                xup[imax] = self.bounds[0][1]
                
                #x[imin] = self.bounds[0][0] - alpha * d[imin]
                #x[imax] = self.bounds[0][1] - alpha * d[imax]
                #np.clip(xup, self.bounds[0][0], self.bounds[0][1], out=xup)
                
                if model == "1/c2":
                    print(f"Maximum velocity: {np.sqrt(1/np.max(xup))}", flush=True)
                    print(f"Minimum velocity: {np.sqrt(1/np.min(xup))}", flush=True)
                elif model == "1/c":
                    print(f"Maximum velocity: {1/np.max(xup)}", flush=True)
                    print(f"Minimum velocity: {1/np.min(xup)}", flush=True)
                else:
                    print(f"Maximum velocity: {np.max(xup)}", flush=True)
                    print(f"Minimum velocity: {np.min(xup)}", flush=True)


                # Check if model perturbation is significant (avoid local minimum).
                # Using 1e7. This controls the accuracy, 1e12 provides low accuracy,
                # 1e7 moderate, and 10 extremely high (scipy Notes).
                epsmch = np.finfo(float).eps
                factr = 1e7*epsmch
                # This condition will need to be improved cause it only applies for the acoustic case
                if (
                    (model == "c" and (((x - xup) / max(np.max(abs(x)), np.max(abs(xup)), 1)) <= factr).all())
                    or (model == "1/c" and (((1 / x - 1 / xup) / max(np.max(abs(1 / x)), np.max(abs(1 / xup)), 1)) <= factr).all())
                    or (model == "1/c2" and (((np.sqrt(1 / x) - np.sqrt(1 / xup)) / max(np.max(abs(np.sqrt(1 / x))), np.max(abs(np.sqrt(1 / xup))), 1)) <= factr).all())
                ):
                    print( "*"*50, flush=True )
                    print("WARNING: model perturbation is too small, stopping.", flush=True)

                    success = 1

                    break

                #np.clip(x, self.bounds[0][0] - alpha * d, self.bounds[0][1] - alpha * d, out=x)
                Jup = self.func( x, *args, alpha=alpha )

                if lineSearchIteration < self.minls and lineSearchIteration < self.maxls:
                    # Adapt alpha depending on Jup and J values
                    if Jup >= J:
                        # Lower alpha
                        alpha *= self.p2

                    else:
                        # Increase alpha
                        alpha *= self.p1

                    lineSearchIteration += 1

                # Min and max number of linesearch iter reached (only when minls == maxls)
                elif lineSearchIteration == self.minls and lineSearchIteration == self.maxls:
                    lineSearchIteration += 1

                    print( "*"*50, flush=True )
                    print( f"FWI iteration {iter} - Max number of linesearch iterations reached \n Last {self.nbCFcheck} full cost functions computed : {checkJ}\n", flush=True )

                    if not all( checkJ ):
                        if Jup >= J:
                            alpha *= 0.1 * self.p2**self.maxls
                            xup = x + alpha * d

                            # Constraining model to bounds
                            xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                        x = xup

                        if self.callback is not None:
                            self.callback( x )

                        iter += 1
                        checkJ[iter % self.nbCFcheck] = Jup

                        break

                    # If checkJ is completely filled
                    else:
                        mean = np.mean( checkJ )
                        # Change the frequency regardless of the value of Jup
                        # If Jup has increased, keep last model, otherwise update model
                        if Jup >= 0.95 * mean:
                            success = 1

                            break

                        else:
                            x = xup

                            if self.callback is not None:
                                self.callback( x )

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                # Minimum number of linesearch iterations reached
                elif lineSearchIteration >= self.minls and lineSearchIteration < self.maxls:
                    lineSearchIteration += 1

                    if not all( checkJ ):
                        if Jup < J:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            alpha *= self.p2

                    else:
                        mean = np.mean( checkJ )

                        if Jup >= 0.95 * mean and Jup <= 1.05 * mean:
                            success = 1

                            break

                        elif Jup < 0.95 * mean:
                            x = xup

                            if self.callback is not None:
                                self.callback(x)

                            iter += 1
                            checkJ[iter % self.nbCFcheck] = Jup

                            break

                        else:
                            alpha *= self.p2

                elif lineSearchIteration >= self.maxls:
                    alpha *= 0.1 * self.p2**self.maxls

                    xup = x + alpha * d

                    # Constraining to bounds
                    xup = np.clip(xup, self.bounds[0][0], self.bounds[0][1])

                    x = xup

                    if self.callback is not None:
                        self.callback( x )

                    iter += 1
                    checkJ[iter % self.nbCFcheck] = Jup

                    break

                # Extra criteria to avoid dwelling on a local minimum
                if np.abs(Jup - J) < 1e-8:
                    print( "*"*50, flush=True )
                    print("Cost function does not change, stopping optimization.", flush=True)

                    success = 1

                    break

            # Ensure a minimum amount of iterations
            if iter < self.miniter:
                if success:
                    success = 0
                    iter += 1

        return x



    def WolfeConditions(self, J, Jup, d, gradJ):
        """
        Check Wolfe conditions

        Parameters
        -----------
            J : float
                Value of the cost function
            Jup : float
                Value of the cost function for perturbed model
            d : array
                Descent direction

        Returns
        --------
            bool :
                0 if conditions not verified, 1 otherwise
        """
        succesArmijo = self.armijoCondition(J, Jup, d, gradJ)

        if succesArmijo:
            succesGoldstein = self.goldsteinCondition(J, Jup, d, gradJ)

            if succesGoldstein:
                return 1
            else:
                self.alpha *= self.p1
                return 0
        else:
            self.alpha *= self.p2
            return 0


    def armijoCondition(self, J, Jm, d, gradJ):
        """
        Check Armijo condition

        Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm <= J + self.k1 * self.alpha * np.dot(d, gradJ))

        return success


    def goldsteinCondition(self, J, Jm, d, gradJ):
        """
        Check Goldstein condition

       Parameters
        ----------
            J : float
                Value of the cost function
            Jm : float
                Value of the cost function for perturbed model
            d : array
                Descent direction
            gradJ : array
                Gradient of the cost function

        Returns
        --------
            success : bool
                0 if test fails, 1 if it passes
        """
        success = (Jm >= J + self.k2 * self.alpha * np.dot(d, gradJ))

        return success
