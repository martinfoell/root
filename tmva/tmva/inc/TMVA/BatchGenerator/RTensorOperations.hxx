// Author: Martin Føll, University of Oslo (UiO) & CERN 10/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RTENSOROPERATIONS
#define TMVA_RTENSOROPERATIONS

#include <vector>
#include <random>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RChunkLoaderFunctor
\ingroup tmva
\brief Loading chunks made in RChunkLoader into tensors from data from RDataFrame.
*/

class RTensorOperations {
private:
   // clang-format on   
   bool fShuffle;
   std::size_t fSetSeed;   
public:
   RTensorOperations(bool shuffle = true, const std::size_t setSeed = 0)
      : fShuffle(shuffle),
        fSetSeed(setSeed)
   {
      std::cout << "Shuffling here" << std::endl;
   }

  void ShuffleTensor(TMVA::Experimental::RTensor<float> &ShuffledTensor, TMVA::Experimental::RTensor<float> &Tensor ) {
    std::random_device rd;
    std::mt19937 g;

    if (fSetSeed == 0) {
      g.seed(rd());
    } else {
      g.seed(fSetSeed);
    }

    std::size_t rows = Tensor.GetShape()[0];
    std::size_t cols = Tensor.GetShape()[1];

    ShuffledTensor.Resize({rows, cols});
    // make an identity permutation map
    std::vector<Long_t> indices(rows);

    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i] = i;
    }

    // shuffle the identity permutation to create a new permutation
    if (fShuffle) {
      std::shuffle(indices.begin(), indices.end(), g);
    }


    // shuffle data in RTensor with the permutation map defined above
    for (std::size_t i = 0; i < rows; i++) {
      std::copy(Tensor.GetData() + indices[i] * cols,
                Tensor.GetData() + (indices[i] + 1) * cols,
                ShuffledTensor.GetData() + i * cols);
    }
  }

   void ConcatinateTwoTensors(TMVA::Experimental::RTensor<float> &MergeTensor, TMVA::Experimental::RTensor<float> &Tensor1, TMVA::Experimental::RTensor<float> &Tensor2) {
      std::size_t rows1 = Tensor1.GetShape()[0];
      std::size_t rows2 = Tensor2.GetShape()[0];
      std::size_t cols = Tensor1.GetShape()[1];
      
      std::size_t index = 0;
      for (std::size_t i = 0; i < rows1; i++) {
         std::copy(Tensor1.GetData() + i * cols, Tensor1.GetData() + (i + 1) * cols,
                   MergeTensor.GetData() + index * cols);
         index++;
      }

      for (std::size_t i = 0; i < rows2; i++) {
         std::copy(Tensor2.GetData() + i * cols, Tensor2.GetData() + (i + 1) * cols,
                   MergeTensor.GetData() + index * cols);
         index++;
      }
   }

   void ConcatinateTensors(TMVA::Experimental::RTensor<float> &ConcatTensor, std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
      std::size_t index = 0;
      for (std::size_t i = 0; i < Tensors.size(); i++) {
         std::size_t rows = Tensors[i].GetShape()[0];
         std::size_t cols = Tensors[i].GetShape()[1];
         for (std::size_t j = 0; j < rows; j++) {
            std::copy(Tensors[i].GetData() + j * cols, Tensors[i].GetData() + (j + 1) * cols,
                      ConcatTensor.GetData() + index * cols);
            index++;
         }
      }
   }
   
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RTENSOROPERATIONS
