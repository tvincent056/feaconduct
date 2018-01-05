#ifndef FLAGBYZ_H_
#define FLAGBYZ_H_

#include "libmesh/mesh_refinement.h"

namespace libMesh
{

class FlagByZ: public MeshRefinement::ElementFlagging
{
public:
    FlagByZ(Real coarsen_z, Real refine_z, MeshBase & mesh):
        coarsen_z_(coarsen_z), refine_z_(refine_z), mesh_(mesh) {}
    virtual ~FlagByZ(){}

    virtual void flag_elements()
    {

        MeshBase::element_iterator       e_it  =
          mesh_.active_elements_begin();
        const MeshBase::element_iterator e_end =
          mesh_.active_elements_end();
        for (; e_it != e_end; ++e_it)
        {
            Elem * elem           = *e_it;
            const dof_id_type id = elem->id();

            Point centroid = elem->centroid();

            if (centroid(2) < coarsen_z_)
                elem->set_refinement_flag(Elem::COARSEN);
            else if (centroid(2) >= refine_z_)
            {
                elem->set_refinement_flag(Elem::REFINE);
            }
        }
    }

protected:
    Real coarsen_z_;
    Real refine_z_;
    MeshBase & mesh_;
};

} //end namespace libMesh
#endif
