#ifndef GITRM_BOUNDARY_HPP
#define GITRM_BOUNDARY_HPP

#include "GitrmMesh.hpp"

/** Storing boundary fields, and extracted separate partial boundary
 * surface mesh for distance-to-boundary calculation.
 * The data from the full mesh is extracted and stored per element
 * of picpart. The surface model interaction handling is not done here.
 */

class GitrmMeshFaceFields;
class GitrmBoundary {

public:

  // To store boundary fields: 
  // 1. BDRY: As separate data of only used boundary faces; no tags; mesh extracted.
  // 2. PICPART: On the whole picpart mesh face as tags; data on non-boundary
  //    faces not used. This case may be used for full-buffer mesh.
  // In either case the data is stored per element of the picpart.
  enum FaceFieldFillCoverage {
    INVALID = 0,
    BDRY = 1,
    PICPART = 2
  };

  GitrmBoundary(GitrmMesh*, FaceFieldFillCoverage);

  GitrmMesh* getGitrmMesh() const { return gm; }
  p::Mesh* getPicparts() const { return picparts; }
  o::Mesh* getPicpart() const { return mesh; }
  o::Mesh* getMesh() const { return mesh; }
  o::Mesh* getFullMesh() const { return fullMesh; }

  bool selectMeshData(o::Reals* coords, o::LOs* face_verts,
   o::Read<o::I8>* exposed, o::LO& nf);

  /** Calculated fields on Boundary faces in preprocessing stage
   * This can be used for separate extracted mesh bdry faces, or 
   * all faces of picpart or all faces of the full mesh
   */
  bool calculateBdryFaceFields(GitrmMeshFaceFields& ff, int debug=0);
  void calculateBdryData(GitrmMeshFaceFields& ff);

  o::Reals calculateScalarFieldOnBdryFacesFromFile(const std::string& fieldName,
    const std::string& file, Field3StructInput& fs, int debug=0);

  bool storeBdrySurfaceMeshForPicpart(int debug = 0);

  void initBoundaries();
  void storeBdryData();
  void setBoundaryInput();

  void storeBdryFaceFieldsForPicpart(bool pre, int debug=0);

  o::Reals getBxSurfNormAngle() const;
  o::Reals getPotential() const;
  o::Reals getDebyeLength() const;
  o::Reals getLarmorRadius() const;
  o::Reals getChildLangmuirDist() const;
  o::Reals getElDensity() const;
  o::Reals getIonDensity() const;
  o::Reals getElTemp() const;
  o::Reals getIonTemp() const;

  o::Reals getBdryFaceVertCoordsPic() const;
  o::LOs getBdryFaceVertsPic() const;
  o::LOs getBdryFaceIdsPic() const;
  o::LOs getBdryFaceIdPtrsPic() const;

  o::Real getBiasPotential() const { return biasPotential;}

  bool isBiasedSurface() const { return isBiasedSurf;}
  bool usingStoredBdryData() const { return useStoredBdryData; }
  bool isAddingTag(FaceFieldFillCoverage);

  bool isUsingOrigBdryData() const { return useOrigBdryData; }

  o::GOs getOrigGlobalIds() const;
  void testDistanceToBdry(int debug=0);

private:
  GitrmMesh* gm;
  p::Mesh* picparts;
  o::Mesh* mesh;
  o::Mesh* fullMesh;
  FaceFieldFillCoverage coverage;

  //on picpart 
  o::Reals bdryFaceVertCoordsPic;
  o::LOs bdryFaceVertsPic;
  o::LOs bdryFaceIdsPic;
  o::LOs bdryFaceIdPtrsPic;

  o::Reals angleBdryBfield;
  o::Reals potential;
  o::Reals debyeLength;
  o::Reals larmorRadius;
  o::Reals childLangmuirDist;
  o::Reals bdryFlux;
  o::Reals impact;
  
  o::Reals elDensity;
  o::Reals elTemp;
  o::Reals ionDensity;
  o::Reals ionTemp;

  bool stored = false;

  /** Original bdry data means that loaded from the file. This is set in the case
   * where the picpart has all the elements of the original fullMesh, since
   * in which case the re-numbering is not done in picpart. 
   */
  bool useOrigBdryData = false;

  /** storedBdryData is copied to fit the re-numbered core elements */
  bool useStoredBdryData = false;

  o::Real biasPotential = 0;
  bool isBiasedSurf = false;
  int rank = -1;
  //bool calcEfieldUsingD2Bdry = false;
  /** If bdry field data added as tags on all faces of all elements */
  bool useBdryTagFields = false;
  
  /** original global ids for picpart nelems, over fullMesh elems. Rest are -1 */
  o::GOs origGlobalIds;
};

/** To pass data used on the bdry
 */
struct GitrmMeshFaceFields {
  o::Reals angle;
  o::Reals debyeLength;
  o::Reals larmorRadius;
  o::Reals childLangmuirDist;
  o::Reals bdryFlux;
  o::Reals impact;
  o::Reals potential;
};

#endif
