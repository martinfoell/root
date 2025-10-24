// Author: Martin Føll, University of Oslo (UiO) & CERN 10/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RDATASETLOADER
#define TMVA_RDATASETLOADER

#include <vector>
#include <random>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "TMVA/BatchGenerator/RChunkConstructor.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"

#include "ROOT/RLogger.hxx"

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RDatasetLoaderFunctor
\ingroup tmva
\brief Loading chunks made in RDatasetLoader into tensors from data from RDataFrame.
*/

template <typename... ColTypes>
class RDatasetLoaderFunctor {
   // clang-format on   
   std::size_t fOffset{};
   std::size_t fVecSizeIdx{};
   float fVecPadding{};
   std::vector<std::size_t> fMaxVecSizes{};
   TMVA::Experimental::RTensor<float> &fChunkTensor;

   std::size_t fNumChunkCols;

   int fI;
   int fNumColumns;

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of vectors 
   template <typename T, std::enable_if_t<ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &vec, int i, int numColumns)
   {
      std::size_t max_vec_size = fMaxVecSizes[fVecSizeIdx++];
      std::size_t vec_size = vec.size();
      if (vec_size < max_vec_size) // Padding vector column to max_vec_size with fVecPadding
      {
         std::copy(vec.begin(), vec.end(), &fChunkTensor.GetData()[fOffset + numColumns * i]);
         std::fill(&fChunkTensor.GetData()[fOffset + numColumns * i + vec_size],
                   &fChunkTensor.GetData()[fOffset + numColumns * i + max_vec_size], fVecPadding);
      } else // Copy only max_vec_size length from vector column
      {
         std::copy(vec.begin(), vec.begin() + max_vec_size, &fChunkTensor.GetData()[fOffset + numColumns * i]);
      }
      fOffset += max_vec_size;
   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Copy the content of a column into RTensor when the column consits of single values 
   template <typename T, std::enable_if_t<!ROOT::Internal::RDF::IsDataContainer<T>::value, int> = 0>
   void AssignToTensor(const T &val, int i, int numColumns)
   {
      fChunkTensor.GetData()[fOffset + numColumns * i] = val;
      fOffset++;
      // fChunkTensor.GetData()[numColumns * i] = val;
   }

public:
   RDatasetLoaderFunctor(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t numColumns,
                       const std::vector<std::size_t> &maxVecSizes, float vecPadding, int i)
      : fChunkTensor(chunkTensor), fMaxVecSizes(maxVecSizes), fVecPadding(vecPadding), fI(i), fNumColumns(numColumns)
   {
   }

   void operator()(const ColTypes &...cols)
   {
      fVecSizeIdx = 0;
      (AssignToTensor(cols, fI, fNumColumns), ...);
   }
};

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RDatasetLoader
\ingroup tmva
\brief Building and loading the chunks from the blocks and chunks constructed in RChunkConstructor

In this class the blocks are stiches together to form chunks that are loaded into memory. The blocks used to create each chunk comes from different parts of the dataset. This is achieved by shuffling the blocks before distributing them into chunks. The purpose of this process is to reduce bias during machine learning training by ensuring that the data is well mixed. The dataset is also spit into training and validation sets with the user-defined validation split fraction.
*/

template <typename... Args>
class RDatasetLoader {
private:
   // clang-format on   
   std::size_t fNumEntries;
   std::size_t fChunkSize;
   std::size_t fBlockSize;
   float fValidationSplit;

   std::vector<std::size_t> fVecSizes;
   std::size_t fSumVecSizes;
   std::size_t fVecPadding;
   std::size_t fNumChunkCols;

   std::size_t fNumTrainEntries;
   std::size_t fNumValidationEntries;

   ROOT::RDF::RNode &f_rdf;
   std::vector<std::string> fCols;
   std::size_t fNumCols;
   std::size_t fSetSeed;

   bool fNotFiltered;
   bool fShuffle;

   ROOT::RDF::RResultPtr<std::vector<ULong64_t>> fEntries;

   std::unique_ptr<RChunkConstructor> fTraining;
   std::unique_ptr<RChunkConstructor> fValidation;

public:
   RDatasetLoader(ROOT::RDF::RNode &rdf, std::size_t numEntries,
                ROOT::RDF::RResultPtr<std::vector<ULong64_t>> rdf_entries, const float validationSplit,
                const std::vector<std::string> &cols, const std::vector<std::size_t> &vecSizes = {},
                const float vecPadding = 0.0, bool shuffle = true, const std::size_t setSeed = 0)
      : f_rdf(rdf),
        fNumEntries(numEntries),
        fEntries(rdf_entries),
        fCols(cols),
        fVecSizes(vecSizes),
        fVecPadding(vecPadding),
        fValidationSplit(validationSplit),
        fNotFiltered(f_rdf.GetFilterNames().empty()),
        fShuffle(shuffle),
        fSetSeed(setSeed)
   {
      fNumCols = fCols.size();
      fSumVecSizes = std::accumulate(fVecSizes.begin(), fVecSizes.end(), 0);

      fNumChunkCols = fNumCols + fSumVecSizes - fVecSizes.size();

      // number of training and validation entries after the split
      fNumValidationEntries = static_cast<std::size_t>(fValidationSplit * fNumEntries);
      fNumTrainEntries = fNumEntries - fNumValidationEntries;

   }

   //////////////////////////////////////////////////////////////////////////
   /// \brief Load the nth chunk from the training dataset into a tensor
   /// \param[in] TrainChunkTensor RTensor for the training chunk
   /// \param[in] chunk Index of the chunk in the dataset
   void SplitDataset(TMVA::Experimental::RTensor<float> &TrainDatasetTensor, TMVA::Experimental::RTensor<float> &ValidationDatasetTensor)
   {

      std::random_device rd;
      std::mt19937 g;

      if (fSetSeed == 0) {
         g.seed(rd());
      } else {
         g.seed(fSetSeed);
      }

      // make an identity permutation map        
      std::vector<int> indices(fNumEntries);
      std::iota(indices.begin(), indices.end(), 0);

      // shuffle the identity permutation to create a new permutation         
      if (fShuffle) {
         std::shuffle(indices.begin(), indices.end(), g);
      }
      
      TMVA::Experimental::RTensor<float> Tensor({fNumEntries, fNumChunkCols});

      if (fNotFiltered) {
         RDatasetLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, 0);
         f_rdf.Foreach(func, fCols);
      }

      else {
         std::size_t datasetEntry = 0;
         for (std::size_t j = 0; j < fNumEntries; j++) {
            RDatasetLoaderFunctor<Args...> func(Tensor, fNumChunkCols, fVecSizes, fVecPadding, datasetEntry);
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdf, (*fEntries)[j], (*fEntries)[j + 1]);
            f_rdf.Foreach(func, fCols);
            datasetEntry++;
         }
      }

      TMVA::Experimental::RTensor<float> ShuffledTensor({fNumEntries, fNumChunkCols});            
      
      // shuffle data in RTensor with the permutation map defined above
      for (std::size_t i = 0; i < fNumEntries; i++) {
         std::copy(Tensor.GetData() + indices[i] * fNumChunkCols,
                   Tensor.GetData() + (indices[i] + 1) * fNumChunkCols,
                   ShuffledTensor.GetData() + i * fNumChunkCols);
      }

      TrainDatasetTensor = ShuffledTensor.Slice({{0, fNumTrainEntries}, {0, fNumChunkCols}});
      ValidationDatasetTensor = ShuffledTensor.Slice({{fNumTrainEntries, fNumEntries}, {0, fNumChunkCols}});
      
   }
   
   std::size_t GetNumTrainingChunks() { return fTraining->Chunks; }

   std::size_t GetNumValidationChunks() { return fValidation->Chunks; }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RDATASETLOADER
