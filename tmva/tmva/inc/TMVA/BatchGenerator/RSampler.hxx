// Author: Martin Føll, University of Oslo (UiO) & CERN 10/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RSAMPLER
#define TMVA_RSAMPLER

#include <vector>
#include <random>
#include <algorithm>

#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RVec.hxx"
#include "TMVA/BatchGenerator/RTensorOperations.hxx"
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

class RSampler {
private:
   // clang-format on   
  std::mt19937 fGen;
   bool fShuffle;
   std::size_t fSetSeed;
   float fSamplingStrategy;
   std::size_t fRandomState;
   bool fReplacement;
   bool fLoadEager;

   std::size_t fMajor;
   std::size_t fMinor;
   std::size_t fNumMajor;
   std::size_t fNumMinor;
   std::size_t fNumRMajor;
   std::size_t fNumRMinor;  
   std::size_t fNumRandomUnderSampler;
   std::size_t fNumUnderSampler;
   std::size_t fNumOverSampler;
  
   std::vector<std::size_t> fSamples;
   std::vector<std::size_t> fSamplesCycled;
   std::vector<std::pair<std::size_t, std::size_t>> fSampleIntervals;
   std::unique_ptr<RTensorOperations> fTensorOperations;   
public:
   RSampler(float samplingStrategy, std::size_t randomState = 42, bool replacement = false, bool shuffle = true, const std::size_t setSeed = 0, bool loadEager = false)
      : fSamplingStrategy(samplingStrategy),
        fRandomState(randomState),
        fGen(randomState),
        fReplacement(replacement),
        fShuffle(shuffle),
        fSetSeed(setSeed),
        fLoadEager(loadEager)
   {
      fTensorOperations = std::make_unique<RTensorOperations>(fShuffle, fSetSeed);
   }

   void SampleWithReplacement(std::size_t n_samples, std::size_t max){
    std::uniform_int_distribution<> dist(0, max - 1);
    fSamples.clear();
    fSamples.reserve(n_samples);
    for (std::size_t i = 0; i < n_samples; ++i) {
      std::size_t sample;
      if (fRandomState == 0) {
        sample = i % max;
      }
      
      else {
        sample = dist(fGen);        
      }
      fSamples.push_back(sample);
    }
   }

   void SampleWithoutReplacement(std::size_t n_samples, std::size_t max){
      std::vector<std::size_t> UniqueSamples;
      UniqueSamples.reserve(max);
      fSamples.clear();
      fSamples.reserve(n_samples);   

      for (std::size_t i = 0; i < max; ++i)
         UniqueSamples.push_back(i);
      
      if (fRandomState != 0) {
        std::shuffle(UniqueSamples.begin(), UniqueSamples.end(), fGen);
      }
      
      for (std::size_t i = 0; i < n_samples; ++i) {
         fSamples.push_back(UniqueSamples[i]);
      }
   }
   

   void RandomSampler(TMVA::Experimental::RTensor<float> &SampledTensor, std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
         std::size_t rows = 0;
         for (std::size_t i = 0; i < Tensors.size(); i++) {
            rows += Tensors[i].GetShape()[0];
         }
         std::size_t cols = Tensors[0].GetShape()[1];
         
         TMVA::Experimental::RTensor<float> Tensor({rows, cols});
         fTensorOperations->ConcatinateTensors(Tensor, Tensors);
         
         SampledTensor = TMVA::Experimental::RTensor<float>({rows, cols});
         fTensorOperations->ShuffleTensor(SampledTensor, Tensor);
   }

  std::size_t SetupRandomUnderSampler(std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
      // std::size_t major;
      // std::size_t minor;

      if (Tensors[0].GetShape()[0] > Tensors[1].GetShape()[0]) {
         fMajor = 0;
         fMinor = 1;
      }
      else {
         fMajor = 1;
         fMinor = 0;         
      }

      fNumMajor = Tensors[fMajor].GetShape()[0];
      fNumMinor = Tensors[fMinor].GetShape()[0];
      fNumRMajor = static_cast<std::size_t>(fNumMinor / fSamplingStrategy);      
      fNumUnderSampler = fNumMinor + fNumRMajor;

      return fNumUnderSampler;
  }

  std::size_t SetupRandomOverSampler(std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
      // std::size_t major;
      // std::size_t minor;

      if (Tensors[0].GetShape()[0] > Tensors[1].GetShape()[0]) {
         fMajor = 0;
         fMinor = 1;
      }
      else {
         fMajor = 1;
         fMinor = 0;         
      }

      fNumMajor = Tensors[fMajor].GetShape()[0];
      fNumMinor = Tensors[fMinor].GetShape()[0];
      fNumRMinor = static_cast<std::size_t>(fSamplingStrategy * fNumMajor);      
      fNumOverSampler = fNumMajor + fNumRMinor;

      return fNumOverSampler;
  }
  
   void RandomUnderSampler(TMVA::Experimental::RTensor<float> &ShuffledSampledTensor, std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
      
      if (fReplacement) {
         SampleWithReplacement(fNumRMajor, fNumMajor);         
      }
      
      else {
         SampleWithoutReplacement(fNumRMajor, fNumMajor);
      }
      
      std::size_t cols = Tensors[0].GetShape()[1];
      ShuffledSampledTensor = TMVA::Experimental::RTensor<float>({fNumUnderSampler, cols});
      TMVA::Experimental::RTensor<float> SampledTensor({fNumUnderSampler, cols});
      TMVA::Experimental::RTensor<float> UnderSampledTensor({fNumRMajor, cols});
      
      std::size_t index = 0;
      for (std::size_t i = 0; i < fNumRMajor; i++) {
         std::copy(Tensors[fMajor].GetData() + fSamples[i] * cols, Tensors[fMajor].GetData() + (fSamples[i]+1) * cols,
                   UnderSampledTensor.GetData() + index * cols);
         index++;
      }

      fTensorOperations->ConcatinateTwoTensors(SampledTensor, UnderSampledTensor, Tensors[fMinor]);
      fTensorOperations->ShuffleTensor(ShuffledSampledTensor, SampledTensor);
   }

   void RandomOverSampler(TMVA::Experimental::RTensor<float> &ShuffledSampledTensor, std::vector<TMVA::Experimental::RTensor<float>> &Tensors) {
      
      SampleWithReplacement(fNumRMinor, fNumMinor);         
      
      std::size_t cols = Tensors[0].GetShape()[1];
      ShuffledSampledTensor = TMVA::Experimental::RTensor<float>({fNumOverSampler, cols});
      TMVA::Experimental::RTensor<float> SampledTensor({fNumOverSampler, cols});
      TMVA::Experimental::RTensor<float> OverSampledTensor({fNumRMinor, cols});
      
      std::size_t index = 0;
      for (std::size_t i = 0; i < fNumRMinor; i++) {
         std::copy(Tensors[fMinor].GetData() + fSamples[i] * cols, Tensors[fMinor].GetData() + (fSamples[i]+1) * cols,
                   OverSampledTensor.GetData() + index * cols);
         index++;
      }

      fTensorOperations->ConcatinateTwoTensors(SampledTensor, OverSampledTensor, Tensors[fMajor]);
      fTensorOperations->ShuffleTensor(ShuffledSampledTensor, SampledTensor);
   }
  
      // else if (fNumDataFrames > 1) {
      //   std::size_t lastBatch = 1;
        
      //   std::size_t n_samples = static_cast<std::size_t>(fSampleStrategy * fTrainingTensors[0].GetShape()[0]);           
      //   TMVA::Experimental::RTensor<float> UnderSampledDataset({n_samples, fNumDatasetCols});
      //   fRSampler->RandomUnderSample(UnderSampledDataset, fTrainingTensors[0]);

      //   std::size_t ConcatDatasetSize = n_samples + fTrainingTensors[1].GetShape()[0];
      //   TMVA::Experimental::RTensor<float> ConcatDataset({ConcatDatasetSize, fNumDatasetCols});
      //   // fTensorOperations->ConcatinateTwoTensors(ConcatDataset, UnderSampledDataset, fTrainingTensors[1]);
      //   std::vector<TMVA::Experimental::RTensor<float>> DatasetTensors = {UnderSampledDataset, fTrainingTensors[1]};
      //   fTensorOperations->ConcatinateTensors(ConcatDataset, DatasetTensors);
      //   // MergeTensors(ConcatDataset, UnderSampledDataset, fTrainingTensors[1]);
        
      //   TMVA::Experimental::RTensor<float> ShuffleConcatDataset({ConcatDatasetSize, fNumDatasetCols});
      //   fTensorOperations->ShuffleTensor(ShuffleConcatDataset, ConcatDataset);

      //   std::size_t LeftoverTrainingBatchSize = ConcatDatasetSize % fBatchSize;
      //   fSampleTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols);
      //   fSampleTrainingBatchLoader->CreateBatches(ShuffleConcatDataset, 1, LeftoverTrainingBatchSize, fDropRemainder, fQueue);
      //   auto batchQueue = fSampleTrainingBatchLoader->GetNumBatchQueue();
      //   std::cout << batchQueue << std::endl;
      //   std::cout << "UnderSample (" << UnderSampledDataset.GetShape()[0] << ") " << UnderSampledDataset << std::endl;
      //   std::cout << "Concat (" << ConcatDataset.GetShape()[0] << ") " << ConcatDataset << std::endl;
      //   std::cout << "ShuffleConcat (" << ShuffleConcatDataset.GetShape()[0] << ") " << ShuffleConcatDataset << std::endl;        
        
      //   // }
      //   // std::exit(0);
      //   // std::vector<TMVA::Experimental::RTensor<float>> TrainingBatches = fBatchLoader->GetTrainBatches();
      //   //     std::cout << TrainingBatches.size() << std::endl; 
        
      // }


   //      else {
   //        fChunkLoader->CreateTrainingChunksIntervals();
   //        fTrainingEpochActive = true;
   //        fTrainingChunkNum = 0;
   //        fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
   //        std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
   //        fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fLeftoverTrainingBatchSize,
   //                                                 fDropRemainder, fQueue);
   //        // fChunkTrainingBatchLoader->FillBatchesInQueue();
   //        fTrainingChunkNum++;

   //      }

   //    if (fNumDataFrames == 1) {
   //    }

   //    else if (fNumDataFrames > 1) {
   //      std::size_t lastBatch = 1;

   //      std::size_t n_samples = static_cast<std::size_t>(fSampleStrategy * fTrainingTensors[0].GetShape()[0]);           
   //      TMVA::Experimental::RTensor<float> UnderSampledDataset({n_samples, fNumDatasetCols});
   //      fRSampler->RandomUnderSample(UnderSampledDataset, fTrainingTensors[0]);

   //      std::size_t ConcatDatasetSize = n_samples + fTrainingTensors[1].GetShape()[0];
   //      TMVA::Experimental::RTensor<float> ConcatDataset({ConcatDatasetSize, fNumDatasetCols});
   //      // fTensorOperations->ConcatinateTwoTensors(ConcatDataset, UnderSampledDataset, fTrainingTensors[1]);
   //      std::vector<TMVA::Experimental::RTensor<float>> DatasetTensors = {UnderSampledDataset, fTrainingTensors[1]};
   //      fTensorOperations->ConcatinateTensors(ConcatDataset, DatasetTensors);
   //      // MergeTensors(ConcatDataset, UnderSampledDataset, fTrainingTensors[1]);
        
   //      TMVA::Experimental::RTensor<float> ShuffleConcatDataset({ConcatDatasetSize, fNumDatasetCols});
   //      fTensorOperations->ShuffleTensor(ShuffleConcatDataset, ConcatDataset);

   //      std::size_t LeftoverTrainingBatchSize = ConcatDatasetSize % fBatchSize;
   //      fSampleTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols);
   //      fSampleTrainingBatchLoader->CreateBatches(ShuffleConcatDataset, 1, LeftoverTrainingBatchSize, fDropRemainder, fQueue);
   //      auto batchQueue = fSampleTrainingBatchLoader->GetNumBatchQueue();
   //      std::cout << batchQueue << std::endl;
   //      std::cout << "UnderSample (" << UnderSampledDataset.GetShape()[0] << ") " << UnderSampledDataset << std::endl;
   //      std::cout << "Concat (" << ConcatDataset.GetShape()[0] << ") " << ConcatDataset << std::endl;
   //      std::cout << "ShuffleConcat (" << ShuffleConcatDataset.GetShape()[0] << ") " << ShuffleConcatDataset << std::endl;        

      
   // }

   // void CycleSortKeepDuplicates() {
   //    std::map<std::size_t, std::size_t> counts;
   //    for (std::size_t x : fSamples) counts[x]++;

   //    for (const auto& [num, count] : counts) {
   //      fSamplesCycled.push_back(num);
   //    }

   //    for (const auto& [num, count] : counts) {
   //       for (std::size_t i = 1; i < count; ++i) {
   //          fSamplesCycled.push_back(num);
   //       }
   //    }
   // }

   // void FindContinuousIntervals() {
   //    std::size_t start = fSamples[0];
   //    std::size_t end = fSamples[0] + 1;

   //    for (std::size_t i = 1; i < fSamples.size(); ++i) {
   //       if (fSamples[i] == end) {
   //          end = fSamples[i] + 1;
   //       } else if (fSamples[i] != end) {
   //          fSampleIntervals.emplace_back(start, end);
   //          start = fSamples[i];
   //          end = fSamples[i] + 1;
   //       }
   //    }
   //    fSampleIntervals.emplace_back(start, end);
   // }


   // std::vector<std::size_t> GetWithoutReplacementSamples(std::size_t n_samples, std::size_t max) {
   //    GeneratorWithoutReplacement(n_samples, max);
   //    return fSamples;
   // }

   // std::vector<std::size_t> GetWithReplacementSamples(std::size_t n_samples, std::size_t max) {
   //    GeneratorWithReplacement(n_samples, max);
   //    return fSamples;
   // }

   // std::vector<std::pair<std::size_t, std::size_t>> GetSampleIntervals() {
   //    FindContinuousIntervals();
   //    return fSampleIntervals;
   // }

   // std::vector<std::size_t> GetSamplesCycled() {
   //    CycleSortKeepDuplicates();
   //    return fSamplesCycled;
   // }
   
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA
#endif // TMVA_RSAMPLER
