#include <iostream>
#include <algorithm>
#include <sstream>
#include <math.h>

#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_refinement.h"
#include "libmesh/gmv_io.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/dof_map.h"
#include "libmesh/sparse_matrix.h"
#include "libmesh/numeric_vector.h"
#include "libmesh/dense_matrix.h"
#include "libmesh/dense_vector.h"
#include "libmesh/exodusII_io.h"

#include "libmesh/linear_implicit_system.h"
#include "libmesh/transient_system.h"

#include "libmesh/elem.h"

using namespace libMesh;

const boundary_id_type BOTTOM_BOUNDARY = 1;
const boundary_id_type TOP_BOUNDARY = 2;

const std::string TOP_BOUNDARY_TYPE = "moving_heat_flux";
const std::string BOTTOM_BOUNDARY_TYPE = "prescribed_temperature";

// Function for heat flux boundary condition
Real heat_flux_bc(const Real x, const Real y, const Real t);

// Function for heat sink (constant temperature) boundary condition
Real temperature_bc(const Real x, const Real y, const Real t);

Real moving_heat_flux(const Real x, const Real y, const Real t);

// Function for matrix assembly
void assemble_cd (EquationSystems & es,
                  const std::string & system_name);

// System initialization/initial conditions
void init_cd (EquationSystems & es,
              const std::string & system_name);

Number initial_value (const Point & p,
                    const Parameters & parameters,
                    const std::string &,
                    const std::string &);

void add_bc(DenseMatrix<Number>& Ke, DenseVector<Number>& Fe,
     const std::vector<Real>& JxW_face, const std::vector<std::vector<Real> >& psi,
     const QGauss& qface, const std::vector<Point>& qface_points,
     const std::string& boundary_type, const Real time, const Real dt);

void add_dirichlet_bc(DenseMatrix<Number>& Ke, DenseVector<Number>& Fe,
     const std::vector<Real>& JxW_face, const std::vector<std::vector<Real> >& psi,
     const QGauss& qface, const std::vector<Point>& qface_points, const Real time, const Real dt);

 void add_neumann_bc(DenseVector<Number>& Fe, const std::vector<Real>& JxW_face,
       const std::vector<std::vector<Real> >& psi, const QGauss& qface,
       const std::vector<Point>& qface_points, const Real time, const Real dt, Real fptr(const Real x, const Real y, const Real t));

// We can now begin the main program.  Note that this
// example will fail if you are using complex numbers
// since it was designed to be run only with real numbers.
int main (int argc, char ** argv)
{
  // Initialize libMesh.
  LibMeshInit init (argc, argv);

  // This example requires a linear solver package.
  libmesh_example_requires(libMesh::default_solver_package() != INVALID_SOLVER_PACKAGE,
                           "--enable-petsc, --enable-trilinos, or --enable-eigen");

  // This example requires Adaptive Mesh Refinement support - although
  // it only refines uniformly, the refinement code used is the same
  // underneath
#ifndef LIBMESH_ENABLE_AMR
  libmesh_example_requires(false, "--enable-amr");
#else

  // Skip this 2D example if libMesh was compiled as 1D-only.
  libmesh_example_requires(2 <= LIBMESH_DIM, "2D support");

  // Read the mesh from file.  This is the coarse mesh that will be used
  // in example 10 to demonstrate adaptive mesh refinement.  Here we will
  // simply read it in and uniformly refine it 5 times before we compute
  // with it.
  //
  // Create a mesh object, with dimension to be overridden later,
  // distributed across the default MPI communicator.
  Mesh mesh(init.comm());

  mesh.read ("mesh.xda");

  // Create a MeshRefinement object to handle refinement of our mesh.
  // This class handles all the details of mesh refinement and coarsening.
  MeshRefinement mesh_refinement (mesh);

  // Uniformly refine the mesh 5 times.
  mesh_refinement.uniformly_refine (5);

  // Print information about the mesh to the screen.
  mesh.print_info();

  // Create an equation systems object.
  EquationSystems equation_systems (mesh);

  // Add a transient system to the EquationSystems
  // object named "Heat-Transfer".
  TransientLinearImplicitSystem & system =
    equation_systems.add_system<TransientLinearImplicitSystem> ("Heat-Transfer");

  // Adds the variable "T" to "Heat-Transfer".  "T"
  // will be approximated using first-order approximation.
  system.add_variable ("T", FIRST);

  // Give the system a pointer to the matrix assembly
  // and initialization functions.
  system.attach_assemble_function (assemble_cd);
  system.attach_init_function (init_cd);

  // Initialize the data structures for the equation system.
  equation_systems.init ();

  // Prints information about the system to the screen.
  equation_systems.print_info();

  // Write out the initial conditions.
#ifdef LIBMESH_HAVE_EXODUS_API
  // If Exodus is available, we'll write all timesteps to the same file
  // rather than one file per timestep.
  std::string exodus_filename = "heat_transfer.e";
  ExodusII_IO(mesh).write_equation_systems (exodus_filename, equation_systems);
#else
  GMVIO(mesh).write_equation_systems ("out_000.gmv", equation_systems);
#endif

  // The Heat-Transfer system requires that we specify
  // the density, specific heat and thermal conductivity.  We will
  // specify them as Real data types and then use the Parameters
  // object to pass them to the assemble function.
  // equation_systems.parameters.set<Real>("density") = 2800;
  // equation_systems.parameters.set<Real>("specific_heat") = 910;
  // equation_systems.parameters.set<Real>("thermal_conductivity") = 250;
  equation_systems.parameters.set<Real>("density") = 8000;
  equation_systems.parameters.set<Real>("specific_heat") = 500;
  equation_systems.parameters.set<Real>("thermal_conductivity") = 16;

  // Solve the system "Heat-Transfer".  This will be done by
  // looping over the specified time interval and calling the
  // solve() member at each time step.  This will assemble the
  // system and call the linear solver.
  const Real dt = 0.02;
  system.time = 0.;

  for (unsigned int t_step = 0; t_step < 500; t_step++)
    {
      // Incremenet the time counter, set the time and the
      // time step size as parameters in the EquationSystem.
      system.time += dt;

      equation_systems.parameters.set<Real> ("time") = system.time;
      equation_systems.parameters.set<Real> ("dt")   = dt;

      // A pretty update message
      libMesh::out << " Solving time step ";

      // Do fancy zero-padded formatting of the current time.
      {
        std::ostringstream out;

        out << std::setw(2)
            << std::right
            << t_step
            << ", time="
            << std::fixed
            << std::setw(6)
            << std::setprecision(3)
            << std::setfill('0')
            << std::left
            << system.time
            <<  "...";

        libMesh::out << out.str() << std::endl;
      }

      // At this point we need to update the old
      // solution vector.  The old solution vector
      // will be the current solution vector from the
      // previous time step.  We will do this by extracting the
      // system from the EquationSystems object and using
      // vector assignment.  Since only TransientSystems
      // (and systems derived from them) contain old solutions
      // we need to specify the system type when we ask for it.
      *system.old_local_solution = *system.current_local_solution;

      // Assemble & solve the linear system
      equation_systems.get_system("Heat-Transfer").solve();

      // Output evey 10 timesteps to file.
      if ((t_step+1)%10 == 0)
        {

#ifdef LIBMESH_HAVE_EXODUS_API
          ExodusII_IO exo(mesh);
          exo.append(true);
          exo.write_timestep (exodus_filename, equation_systems, t_step+1, system.time);
#else
          std::ostringstream file_name;

          file_name << "out_"
                    << std::setw(3)
                    << std::setfill('0')
                    << std::right
                    << t_step+1
                    << ".gmv";


          GMVIO(mesh).write_equation_systems (file_name.str(),
                                              equation_systems);
#endif
        }
    }
#endif // #ifdef LIBMESH_ENABLE_AMR

  // All done.
  return 0;
}

// We now define the function which provides the
// initialization routines for the "Heat-Transfer"
// system.  This handles things like setting initial
// conditions and boundary conditions.
void init_cd (EquationSystems & es,
              const std::string & libmesh_dbg_var(system_name))
{
  // It is a good idea to make sure we are initializing
  // the proper system.
  libmesh_assert_equal_to (system_name, "Heat-Transfer");

  // Get a reference to the Heat-Transfer system object.
  TransientLinearImplicitSystem & system =
    es.get_system<TransientLinearImplicitSystem>("Heat-Transfer");

  // Project initial conditions at time 0
  es.parameters.set<Real> ("time") = system.time = 0;

  system.project_solution(initial_value, libmesh_nullptr, es.parameters);
}



// Now we define the assemble function which will be used
// by the EquationSystems object at each timestep to assemble
// the linear system for solution.
void assemble_cd (EquationSystems & es,
                  const std::string & system_name)
{
    // Ignore unused parameter warnings when !LIBMESH_ENABLE_AMR.
    libmesh_ignore(es);
    libmesh_ignore(system_name);

#ifdef LIBMESH_ENABLE_AMR
    // It is a good idea to make sure we are assembling
    // the proper system.
    libmesh_assert_equal_to (system_name, "Heat-Transfer");

    // Get a constant reference to the mesh object.
    const MeshBase & mesh = es.get_mesh();

    // The dimension that we are running
    const unsigned int dim = mesh.mesh_dimension();

    // Get a reference to the Heat-Transfer system object.
    TransientLinearImplicitSystem & system =
        es.get_system<TransientLinearImplicitSystem> ("Heat-Transfer");

    // Get a constant reference to the Finite Element type
    // for the first (and only) variable in the system.
    FEType fe_type = system.variable_type(0);

    // Build a Finite Element object of the specified type.  Since the
    // FEBase::build() member dynamically creates memory we will
    // store the object as a UniquePtr<FEBase>.  This can be thought
    // of as a pointer that will clean up after itself.
    UniquePtr<FEBase> fe      (FEBase::build(dim, fe_type));
    UniquePtr<FEBase> fe_face (FEBase::build(dim, fe_type));

    // A Gauss quadrature rule for numerical integration.
    // Let the FEType object decide what order rule is appropriate.
    QGauss qrule (dim,   fe_type.default_quadrature_order());
    QGauss qface (dim-1, fe_type.default_quadrature_order());

    // Tell the finite element object to use our quadrature rule.
    fe->attach_quadrature_rule      (&qrule);
    fe_face->attach_quadrature_rule (&qface);

    // Here we define some references to cell-specific data that
    // will be used to assemble the linear system.  We will start
    // with the element Jacobian * quadrature weight at each integration point.
    const std::vector<Real> & JxW      = fe->get_JxW();
    const std::vector<Real> & JxW_face = fe_face->get_JxW();

    // The element shape functions evaluated at the quadrature points.
    const std::vector<std::vector<Real> > & phi = fe->get_phi();
    const std::vector<std::vector<Real> > & psi = fe_face->get_phi();

    // The element shape function gradients evaluated at the quadrature
    // points.
    const std::vector<std::vector<RealGradient> > & dphi = fe->get_dphi();

    // The XY locations of the quadrature points used for face integration
    const std::vector<Point> & qface_points = fe_face->get_xyz();

    const std::vector<Point> & qrule_points = fe->get_xyz();

    // A reference to the DofMap object for this system.  The DofMap
    // object handles the index translation from node and element numbers
    // to degree of freedom numbers.  We will talk more about the DofMap
    // in future examples.
    const DofMap & dof_map = system.get_dof_map();

    // Define data structures to contain the element matrix
    // and right-hand-side vector contribution.  Following
    // basic finite element terminology we will denote these
    // "Ke" and "Fe".
    DenseMatrix<Number> Ke;
    DenseVector<Number> Fe;

    // This vector will hold the degree of freedom indices for
    // the element.  These define where in the global system
    // the element degrees of freedom get mapped.
    std::vector<dof_id_type> dof_indices;

    // Here we extract the velocity & parameters that we put in the
    // EquationSystems object.
    const Real dt = es.parameters.get<Real>("dt");
    const Real thermal_conductivity = es.parameters.get<Real>("thermal_conductivity");
    const Real density = es.parameters.get<Real>("density");
    const Real specific_heat = es.parameters.get<Real>("specific_heat");

    // Now we will loop over all the elements in the mesh that
    // live on the local processor. We will compute the element
    // matrix and right-hand-side contribution.  Since the mesh
    // will be refined we want to only consider the ACTIVE elements,
    // hence we use a variant of the active_elem_iterator.
    MeshBase::const_element_iterator       el     = mesh.active_local_elements_begin();
    const MeshBase::const_element_iterator end_el = mesh.active_local_elements_end();

    for ( ; el != end_el; ++el)
    {
        // Store a pointer to the element we are currently
        // working on.  This allows for nicer syntax later.
        const Elem * elem = *el;

        // Get the degree of freedom indices for the
        // current element.  These define where in the global
        // matrix and right-hand-side this element will
        // contribute to.
        dof_map.dof_indices (elem, dof_indices);

        // Compute the element-specific data for the current
        // element.  This involves computing the location of the
        // quadrature points (q_point) and the shape functions
        // (phi, dphi) for the current element.
        fe->reinit (elem);

        // Zero the element matrix and right-hand side before
        // summing them.  We use the resize member here because
        // the number of degrees of freedom might have changed from
        // the last element.  Note that this will be the case if the
        // element type is different (i.e. the last element was a
        // triangle, now we are on a quadrilateral).
        Ke.resize (dof_indices.size(),
                   dof_indices.size());

        Fe.resize (dof_indices.size());

        // Now we will build the element matrix and right-hand-side.
        // Constructing the RHS requires the solution and its
        // gradient from the previous timestep.  This myst be
        // calculated at each quadrature point by summing the
        // solution degree-of-freedom values by the appropriate
        // weight functions.
        for (unsigned int qp=0; qp<qrule.n_points(); qp++)
        {
            // Values to hold the old solution & its gradient.
            Number T_old = 0.;
            Gradient grad_T_old;

            // Compute the old solution & its gradient.
            for (std::size_t l=0; l<phi.size(); l++)
            {
                T_old += phi[l][qp]*system.old_solution  (dof_indices[l]);

                // This will work,
                // grad_T_old += dphi[l][qp]*system.old_solution (dof_indices[l]);
                // but we can do it without creating a temporary like this:
                grad_T_old.add_scaled (dphi[l][qp], system.old_solution (dof_indices[l]));
            }

            // Now compute the element matrix and RHS contributions.
            for (std::size_t i=0; i<phi.size(); i++)
            {
                const Number flux = moving_heat_flux (qrule_points[qp](0),
                                                  qrule_points[qp](1),
                                                  system.time);

                // The RHS contribution
                Fe(i) += JxW[qp]*(
                                  // heat capacity matrix term
                                  density*specific_heat*T_old*phi[i][qp] +
                                  -.5*dt*(
                                          // Diffusion term
                                          thermal_conductivity*(grad_T_old*dphi[i][qp]))
                                        //   + flux*phi[i][qp]
                                  );

                for (std::size_t j=0; j<phi.size(); j++)
                {
                    // The matrix contribution
                    Ke(i,j) += JxW[qp]*(
                                        // heat capacity-matrix
                                        density*specific_heat*phi[i][qp]*phi[j][qp] +

                                        .5*dt*(
                                               // Diffusion term
                                              thermal_conductivity*(dphi[i][qp]*dphi[j][qp]))
                                        );
                }
            }
        }

        //apply BCs
        {
            // The following loops over the sides of the element.
            // If the element has no neighbor on a side then that
            // side MUST live on a boundary of the domain.
            for (unsigned int s=0; s<elem->n_sides(); s++)
            {
                if (elem->neighbor_ptr(s) == libmesh_nullptr)
                {
                    fe_face->reinit(elem, s);

                    if (mesh.get_boundary_info().has_boundary_id (elem, s, BOTTOM_BOUNDARY))
                    {
                        add_bc(Ke, Fe, JxW_face, psi, qface, qface_points, BOTTOM_BOUNDARY_TYPE, system.time, dt);
                    }
                    else if(mesh.get_boundary_info().has_boundary_id (elem, s, TOP_BOUNDARY))
                    {
                        add_bc(Ke, Fe, JxW_face, psi, qface, qface_points, TOP_BOUNDARY_TYPE, system.time, dt);
                    }
                    //otherwise a natural zero heat flux BC is applied
                }
            } //end loop over sides
        }

        // If this assembly program were to be used on an adaptive mesh,
        // we would have to apply any hanging node constraint equations
        dof_map.constrain_element_matrix_and_vector (Ke, Fe, dof_indices);

        // The element matrix and right-hand-side are now built
        // for this element.  Add them to the global matrix and
        // right-hand-side vector.  The SparseMatrix::add_matrix()
        // and NumericVector::add_vector() members do this for us.
        system.matrix->add_matrix (Ke, dof_indices);
        system.rhs->add_vector    (Fe, dof_indices);
    }

    // That concludes the system matrix assembly routine.
#endif // #ifdef LIBMESH_ENABLE_AMR
}

void add_bc(DenseMatrix<Number>& Ke, DenseVector<Number>& Fe,
     const std::vector<Real>& JxW_face, const std::vector<std::vector<Real> >& psi,
     const QGauss& qface, const std::vector<Point>& qface_points,
     const std::string& boundary_type, const Real time, const Real dt)
{
    if (boundary_type == "heat_flux")
    {
        add_neumann_bc(Fe, JxW_face, psi, qface, qface_points, time, dt, heat_flux_bc);
    }
    else if (boundary_type == "prescribed_temperature")
    {
        add_dirichlet_bc(Ke, Fe, JxW_face, psi, qface, qface_points, time, dt);
    }
    else if (boundary_type == "moving_heat_flux")
    {
        add_neumann_bc(Fe, JxW_face, psi, qface, qface_points, time, dt, moving_heat_flux);
    }
}

void add_dirichlet_bc(DenseMatrix<Number>& Ke, DenseVector<Number>& Fe,
     const std::vector<Real>& JxW_face, const std::vector<std::vector<Real> >& psi,
     const QGauss& qface, const std::vector<Point>& qface_points, const Real time,
     const Real dt)
{
    const Real penalty = 1.e10;

    for (unsigned int qp=0; qp<qface.n_points(); qp++)
    {
        const Number value_old = temperature_bc (qface_points[qp](0),
                                           qface_points[qp](1),
                                           time - dt);

        const Number value_new = temperature_bc (qface_points[qp](0),
                                           qface_points[qp](1),
                                           time);

        const Number value = 0.5 * (value_old + value_new);

        // RHS contribution
        for (std::size_t i=0; i<psi.size(); i++)
            Fe(i) += penalty*JxW_face[qp]*value*psi[i][qp];

        // Matrix contribution
        for (std::size_t i=0; i<psi.size(); i++)
            for (std::size_t j=0; j<psi.size(); j++)
                Ke(i,j) += penalty*JxW_face[qp]*psi[i][qp]*psi[j][qp];
    }
}

void add_neumann_bc(DenseVector<Number>& Fe, const std::vector<Real>& JxW_face,
      const std::vector<std::vector<Real> >& psi, const QGauss& qface,
      const std::vector<Point>& qface_points, const Real time, const Real dt,
      Real fptr(const Real x, const Real y, const Real t))
{
    for (unsigned int qp=0; qp<qface.n_points(); qp++)
    {
        const Number flux_old = fptr (qface_points[qp](0),
                                      qface_points[qp](1),
                                      time - dt);
        const Number flux_new = fptr (qface_points[qp](0),
                                      qface_points[qp](1),
                                      time);

        const Number flux = 0.5*(flux_old + flux_new);

        // RHS contribution
        for (std::size_t i=0; i<psi.size(); i++)
            Fe(i) += JxW_face[qp]*flux*dt*psi[i][qp];
    }
}

Real heat_flux_bc(const Real x, const Real y, const Real t)
{
    // return 107.;
    // return 1000.;
    return 0.;
    // const Real omega = 2 * M_PI / 20000.;
    // return 10000. * (sin(omega * t) + 1.);
}

Real temperature_bc(const Real x, const Real y, const Real t)
{
    return 293.;
    // const Real omega = 2 * M_PI / 10.;
    // return 334. + 50. * cos(omega*t);
}

Real moving_heat_flux(const Real x, const Real y, const Real t)
{
    Point center;

    const Real tmax = 10.;

    if (t < tmax)
    {
        const Real speed = 8e-3; //mm / s

        const Real trackLength = 8e-3; //mm
        const Real pos = speed * t;
        const Real track = std::floor(pos / trackLength);
        const Real trackPos = (pos - track*trackLength);

        const double sw = fmod(track, 2.);

        center(0) = (1-sw)*(1e-3 + trackPos) + sw * (9e-3 - trackPos);
        center(1) = 1e-3 + track * 0.5e-3;
    }
    else
    {
        return 0;
    }

    Real r = Point(Point(x,y) - center).norm();
    if (r < 0.5e-3)
    {
        return 8.e7;
    }
    else
    {
        return 0.;
    }
}

Number initial_value (const Point & p,
                    const Parameters & parameters,
                    const std::string &,
                    const std::string &)
{
    return 293.;
}
