/*BHEADER**********************************************************************
 * Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
 * Produced at the Lawrence Livermore National Laboratory. Written by 
 * Jacob Schroder, Rob Falgout, Tzanio Kolev, Ulrike Yang, Veselin 
 * Dobrev, et al. LLNL-CODE-660355. All rights reserved.
 * 
 * This file is part of XBraid. For support, post issues to the XBraid Github page.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public
 * License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 ***********************************************************************EHEADER*/

 /**
 * Example:       ex-04.c
 *
 * Interface:     C
 * 
 * Requires:      only C-language support     
 *
 * Compile with:  make ex-04
 *
 * Description:  Solves a simple optimal control problem in time-parallel:
 * 
 *                 min   \int_0^1 u_1(t)^2 + u_2(t)^2 + gamma c(t)^2  dt
 * 
 *                  s.t.  d/dt u_1(t) = u_2(t)
 *                        d/dt u_2(t) = -u_2(t) + c(t)
 * 
 *               with initial condition u_1(0) = 0, u_2(0) = -1
 *               and piecewise constant control c(t).  
 *
 *               Implements a steepest-descent optimization iteration
 *               using fixed step size for design updates.   
 **/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "braid.h"
#include "_braid.h"
#include "braid_test.h"
#include "ex-04-lib.c"


/*--------------------------------------------------------------------------
 * My App and Vector structures
 *--------------------------------------------------------------------------*/

typedef struct _braid_App_struct
{
   int     myid;        /* Rank of the processor */
   double *design;      /* Holds time-dependent design (i.e. control) vector */
   double *gradient;    /* Holds the gradient vector */
   double  gamma;       /* Relaxation parameter for objective function */
   int     ntime;       /* Total number of time-steps */
   double Tfinal;
   double objective; 
   int doadjoint;
   braid_Core core;
} my_App;


/* Define the state vector at one time-step */
typedef struct _braid_Vector_struct
{
   double *values;     /* Holds the R^2 state vector (u_1, u_2) */
   double *valuesbar;     /* Holds the R^2 state vector (u_1, u_2) */

} my_Vector;


/*--------------------------------------------------------------------------
 * Integration routines
 *--------------------------------------------------------------------------*/

int 
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
   int    index;
   double tstart, tstop;
   double design;
   double deltaT;

   /* Get the time-step size */
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
   deltaT = tstop - tstart;

   /* Get the current design at tstart from the app */
   braid_StepStatusGetTIndex(status, &index);
   design = app->design[index];

   /* Take one step forward */
   // printf("Step %f->%f u0=(%f, %f)", tstart, tstop, u->values[0], u->values[1]);
   take_step(u->values, design, deltaT);
   // printf(", u1=(%f,%f)\n", u->values[0], u->values[1]);

   /* Take a step backwards */
   if (app->doadjoint>=1) {   
      // get the state at N-(n+1)
      int FWDid= app->ntime - (index +1);
      braid_BaseVector ubase;
      braid_Vector uFWD=NULL;
      _braid_UGetVectorRef(app->core, 0, FWDid, &ubase);
      if (ubase != NULL) {
         uFWD = ubase->userVector;
      }
      // Update adjoint and design
      // printf("Step (%f->%f)=(%d->%d) ", tstart, tstop, FWDid+1, FWDid); 
      double dPhidp = take_step_diff(u->valuesbar, deltaT);
      double dt_fine = app->Tfinal/ app->ntime;
      // assert(dt_fine == deltaT);
      double dJdp = evalObjectiveT_diff(u->valuesbar, uFWD->values, app->design[FWDid], app->gamma, dt_fine);
      // printf("ubar_out=(%f,%f), dPhi[%d]=%1.8e dJ[%d]=%1.8e \n", u->valuesbar[0], u->valuesbar[1], FWDid, dPhidp, FWDid, dJdp);
      app->gradient[FWDid] = dPhidp + dJdp;
   }

   /* no refinement */
   braid_StepStatusSetRFactor(status, 1);

   return 0;
}   


int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{

   my_Vector *u;

   /* Allocate the vector */
   u = (my_Vector *) malloc(sizeof(my_Vector));
   u->values = (double*) malloc( 2*sizeof(double) );
   u->valuesbar = (double*) malloc( 2*sizeof(double) );

   /* Initialize the primal vector */
   if (t == 0.0)
   {
      u->values[0] = 0.0;
      u->values[1] = -1.0;
   }
   else
   {
      u->values[0] = 0.0;
      u->values[1] = 0.0;
   }

   /* Initialize the adjoint vector */
   // derivative dJdu(T) should here, but Init is not called anymore after second iter...
   {
      u->valuesbar[0] = 0.0;
      u->valuesbar[1] = 0.0;
   }


   *u_ptr = u;

   return 0;
}

int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
   my_Vector *v;

   /* Allocate the vector */
   v = (my_Vector *) malloc(sizeof(my_Vector));
   v->values = (double*) malloc( 2*sizeof(double) );
   v->valuesbar = (double*) malloc( 2*sizeof(double) );

   /* Clone the values */
   v->values[0] = u->values[0];
   v->values[1] = u->values[1];
   v->valuesbar[0] = u->valuesbar[0];
   v->valuesbar[1] = u->valuesbar[1];

   *v_ptr = v;

   return 0;
}


int
my_Free(braid_App    app,
        braid_Vector u)
{
   free(u->values);
   free(u->valuesbar);
   free(u);

   return 0;
}


int
my_Sum(braid_App     app,
       double        alpha,
       braid_Vector  x,
       double        beta,
       braid_Vector  y)
{

   (y->values)[0] = alpha*(x->values)[0] + beta*(y->values)[0];
   (y->values)[1] = alpha*(x->values)[1] + beta*(y->values)[1];
   (y->valuesbar)[0] = alpha*(x->valuesbar)[0] + beta*(y->valuesbar)[0];
   (y->valuesbar)[1] = alpha*(x->valuesbar)[1] + beta*(y->valuesbar)[1];

   return 0;
}


int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
   int i;
   double dot = 0.0;

   for (i = 0; i < 2; i++)
   {
      dot += (u->values)[i]*(u->values)[i];
      dot += (u->valuesbar)[i]*(u->valuesbar)[i];
   }
   *norm_ptr = sqrt(dot);

   return 0;
}


int
my_Access(braid_App          app,
          braid_Vector       u,
          braid_AccessStatus astatus)
{
   int   done, index;
   char  filename[255];
   FILE * file;

   // /* Print solution to file if simulation is over */
   braid_AccessStatusGetDone(astatus, &done);

   // if (done)
   // {
   //    braid_AccessStatusGetTIndex(astatus, &index);
   //    sprintf(filename, "%s.%04d.%03d", "ex-04.out", index, app->myid);
   //    file = fopen(filename, "w");
   //    fprintf(file, "%1.14e, %1.14e\n", (u->values)[0], (u->values)[1]);
   //    fflush(file);
   //    fclose(file);
   // }

   if (done) {
      double deltaT = app->Tfinal/app->ntime;
      double time;
      braid_AccessStatusGetTIndex(astatus, &index);
      braid_AccessStatusGetT(astatus, &time);
      // if (index < app->ntime) {
         double objT = evalObjectiveT(u->values, app->design[index], deltaT, app->gamma);
         // printf("%d, %f, u->values[0]=%f\n", index, time, (u->values)[0]);
         // printf("%d, %f, u->valuesbar[0]=%f\n", index, time, (u->valuesbar)[0]);
         // printf("%d, %f, u->values[0]=%f, objT=%f\n", index, time, (u->values)[0], objT);
         app->objective += objT;
      // }
   }

   return 0;
}


int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  bstatus)
{
   *size_ptr = 4*sizeof(double);
   return 0;
}


int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void               *buffer,
           braid_BufferStatus  bstatus)
{
   double *dbuffer = buffer;
   int i;

   int bufid=0;
   for(i = 0; i < 2; i++)
   {
      dbuffer[bufid] = (u->values)[i]; bufid++;
   }
   for(i = 0; i < 2; i++)
   {
      dbuffer[bufid] = (u->valuesbar)[i]; bufid++;
   }  
   braid_BufferStatusSetSize( bstatus,  4*sizeof(double));

   return 0;
}


int
my_BufUnpack(braid_App           app,
             void               *buffer,
             braid_Vector       *u_ptr,
             braid_BufferStatus  bstatus)
{
   my_Vector *u = NULL;
   double    *dbuffer = buffer;
   int i;

   /* Allocate memory */
   u = (my_Vector *) malloc(sizeof(my_Vector));
   u->values = (double*) malloc( 2*sizeof(double) );
   u->valuesbar = (double*) malloc( 2*sizeof(double) );

   /* Unpack the buffer */
   int bufid=0;
   for(i = 0; i < 2; i++)
   {
      (u->values)[i] = dbuffer[bufid]; bufid++;
   }
   for(i = 0; i < 2; i++)
   {
      (u->valuesbar)[i] = dbuffer[bufid]; bufid++;
   }
   *u_ptr = u;
   return 0;
}

/* Set the gradient to zero */
int 
my_ResetGradient(braid_App app)
{
   int ts;

   for(ts = 0; ts < app->ntime+1; ts++) 
   {
      app->gradient[ts] = 0.0;
   }

   return 0;
}



/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/
int main (int argc, char *argv[])
{

   braid_Core core;
   my_App     *app;
         
   double   tstart, tstop; 
   int      rank, ntime, ts, iter, maxiter, nreq, arg_index;
   double  *design; 
   double  *gradient; 
   double   objective, gamma, stepsize, mygnorm, gnorm;
   double   gtol;
   double   rnorm, rnorm_adj;

   int      max_levels, cfactor, access_level, print_level, braid_maxiter;
   double   braid_tol, braid_adjtol;
   double   dt, h_inv;

   /* Define time domain */
   ntime  = 20;              /* Total number of time-steps */
   tstart = 0.0;             /* Beginning of time domain */
   tstop  = 1.0;             /* End of time domain*/

   /* Define some optimization parameters */
   gamma    = 0.005;         /* Relaxation parameter in the objective function */
   stepsize = 5.0;            /* Step size for design updates */
   maxiter  = 1;           /* Maximum number of optimization iterations */
   gtol     = 1e-6;          /* Stopping criterion on the gradient norm */

   /* Define some Braid parameters */
   max_levels     = 100;
   braid_maxiter  = 10;
   cfactor        = 2;
   braid_tol      = 1.0e-6;
   braid_adjtol   = 1.0e-6;
   access_level   = 1;
   print_level    = 1;
   

   /* Parse command line */
   arg_index = 1;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         printf("\n");
         printf(" Solves a simple optimal control problem in time-serial on [0, 1] \n\n");
         printf("  min   \\int_0^1 u_1(t)^2 + u_2(t)^2 + gamma c(t)^2  dt \n\n");
         printf("  s.t.  d/dt u_1(t) = u_2(t) \n");
         printf("        d/dt u_2(t) = -u_2(t) + c(t) \n\n");
         printf("  -ntime <ntime>          : set num points in time\n");
         printf("  -gamma <gamma>          : Relaxation parameter in the objective function \n");
         printf("  -stepsize <stepsize>    : Step size for design updates \n");
         printf("  -mi <maxiter>           : Maximum number of optimization iterations \n");
         printf("  -ml <max_levels>        : Max number of braid levels \n");
         printf("  -bmi <braid_maxiter>    : Braid max_iter \n");
         printf("  -cf <cfactor>           : Coarsening factor \n");
         printf("  -btol <braid_tol>       : Braid halting tolerance \n");
         printf("  -batol <braid_adjtol>   : Braid adjoint halting tolerance \n");
         printf("  -access <access_level>  : Braid access level \n");
         printf("  -print <print_level>    : Braid print level \n");
         exit(1);
      }
      else if ( strcmp(argv[arg_index], "-ntime") == 0 )
      {
         arg_index++;
         ntime = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-gamma") == 0 )
      {
         arg_index++;
         gamma = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-stepsize") == 0 )
      {
         arg_index++;
         stepsize = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mi") == 0 )
      {
         arg_index++;
         maxiter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ml") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-bmi") == 0 )
      {
         arg_index++;
         braid_maxiter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cfactor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-btol") == 0 )
      {
         arg_index++;
         braid_tol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-batol") == 0 )
      {
         arg_index++;
         braid_adjtol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-access") == 0 )
      {
         arg_index++;
         access_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_level = atoi(argv[arg_index++]);
      }
      else
      {
         printf("ABORTING: incorrect command line parameter %s\n", argv[arg_index]);
         return (0);
      }
   }
   
   /* Initialize optimization */
   // eval J at t=0 AND t=ntime!
   design   = (double*) malloc( (ntime+1)*sizeof(double) );    /* design vector (control c) */
   gradient = (double*) malloc( (ntime+1)*sizeof(double) );    /* gradient vector */
   for (ts = 0; ts < ntime+1; ts++)
   {
      design[ts]   = 0.1;
      gradient[ts] = 0.;
   }
   /* Inverse of reduced Hessian approximation */
   dt    = (tstop - tstart) / ntime;
   h_inv = 1. / ( 2 * dt * (1. + gamma) );

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   
   /* Set up the app structure */
   app = (my_App *) malloc(sizeof(my_App));
   app->myid     = rank;
   app->ntime    = ntime;
   app->design   = design;
   app->gradient = gradient;
   app->gamma    = gamma;
   app->objective = 0.0;
   app->Tfinal = tstop;

   /* Initialize XBraid */
   braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, tstart, tstop, ntime, app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);
   app->core = core;

  /* Initialize XBraid_Adjoint */
   // braid_InitAdjoint( my_ObjectiveT, my_ObjectiveT_diff, my_Step_diff, my_ResetGradient, &core);

   /* Set some XBraid(_Adjoint) parameters */
   braid_SetMaxLevels(core, max_levels);
   braid_SetCFactor(core, -1, cfactor);
   braid_SetAccessLevel(core, access_level);
   braid_SetPrintLevel( core, print_level);       
   braid_SetMaxIter(core, braid_maxiter);
   braid_SetAbsTol(core, braid_tol);
   // braid_SetAbsTolAdjoint(core, braid_adjtol);
   braid_SetSkip(core, 0); // turn off skip of first down cycle.
   braid_SetStorage(core, 0);
   braid_SetFinalFCRelax(core);

   /* Prepare optimization output */
   if (rank == 0)
   {
      printf("\nOptimization:         || r ||        Objective           || Gradient ||\n");
   }

   // Set up all grid points. Here, Running braid without adjoint once. TODO. 
   printf("Initial braid run to set up the grid");
   app->doadjoint = 0;
   app->objective = 0.0;
   braid_Drive(core);
   app->doadjoint = 1;
   my_ResetGradient(app);
   printf("\n\n -- Optim -- \n\n");

   /* Optimization iteration */
   for (iter = 0; iter < maxiter; iter++)
   {
      app->objective = 0.0;
      my_ResetGradient(app);

      // if (iter > 0) {
      // get the state at N
      int FWDid= app->ntime;
      braid_BaseVector ubaseFWD, ubaseBWD;
      braid_Vector uFWD=NULL;
      braid_Vector uBWD=NULL;
      _braid_UGetVectorRef(app->core, 0, FWDid, &ubaseFWD);      // last step
      _braid_UGetVectorRef(app->core, 0, 0, &ubaseBWD); // last first step
      if (ubaseFWD != NULL) {
         uFWD = ubaseFWD->userVector;
      }
      if (ubaseBWD != NULL) {
         uBWD = ubaseBWD->userVector;
      }
      double designi = app->design[FWDid-1];
      // Set adjoint terminal condition
      double dt_fine = app->Tfinal / app->ntime;
      app->gradient[FWDid-1] += evalObjectiveT_diff(uBWD->valuesbar, uFWD->values,designi, app->gamma, dt_fine);
      // printf("Adjoint initial condition: uBWD=(%f,%f)\n", uBWD->valuesbar[0], uBWD->valuesbar[1]);
      // }

      /* Parallel-in-time simulation and gradient computation */
      app->doadjoint=1;
      braid_Drive(core);
      // printf("Objective %f\n", app->objective);
      mygnorm = compute_sqnorm(app->gradient, ntime+1);

      /* Get objective function value */
      // nreq = -1;
      // braid_GetObjective(core, &objective);


      /* Get the state and adjoint residual norms */
      nreq = -1;
      braid_GetRNorms(core, &nreq, &rnorm);
      // braid_GetRNormAdjoint(core, &rnorm_adj);

   
      /* Compute the norm of the gradient */
      MPI_Allreduce(&mygnorm, &gnorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      gnorm = sqrt(gnorm);

      /* Output */
      if (rank == 0)
      {
         printf("Optimization: %3d  %1.8e  %1.14e  %1.14e\n", iter, rnorm, app->objective, gnorm);
      }

      /* Check optimization convergence */
      if (gnorm < gtol)
      {
         printf("Success! \n");
         break;
      }

      /* Preconditioned design update */
      for(ts = 0; ts < ntime+1; ts++) 
      {
         app->design[ts] -= stepsize * h_inv * app->gradient[ts];
      }

   }

   // print gradient
   // printf("Gradient=\n");
   // for (ts=0; ts<ntime+1; ts++){
   //    printf("%d %1.14e\n", ts, app->gradient[ts]);
   // }
   printf("||g||= %1.12e\n", gnorm);
   
#if 0
   //############  FD  ################ 

   double obj_org = app->objective;
   double* grad_org = (double*) malloc( (ntime+1)*sizeof(double) ); 
   for (ts = 0; ts < ntime+1; ts++)
   {
      grad_org[ts] = app->gradient[ts];
   }

   double EPS=1e-6;
   for (int ts=0; ts<ntime+1; ts++){
   // for (int ts=0; ts<1; ts++){
         // p += EPS
         app->design[ts] += EPS;
         app->objective = 0.0;
         my_ResetGradient(app);
         braid_Drive(core);
         double obj_1 = app->objective;
         app->design[ts] -= EPS;

         // p -= EPS
         app->design[ts] -= EPS;
         app->objective = 0.0;
         my_ResetGradient(app);
         braid_Drive(core);
         double obj_2 = app->objective;
         app->design[ts] += EPS;

         // error
         double fd = (obj_1 - obj_2) / (2.*EPS);
         double err = (fd  - grad_org[ts]) / (fd + 1e-16);
         printf("%d: fd=%1.10e, grad=%1.10e, err=%1.10e\n", ts, fd, grad_org[ts], err);

   } 
  #endif
   
   // if (rank == 0) {
   //     /* Write final design to file */
   //     write_design_vec("design", design, ntime);
   // }

   /* Clean up */
   free(design);
   free(gradient);
   free(app);
   
   braid_Destroy(core);
   MPI_Finalize();

   return (0);
}
