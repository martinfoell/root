// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 05/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TMVA_RBATCHGENERATOR
#define TMVA_RBATCHGENERATOR

#include "TMVA/RTensor.hxx"
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "TMVA/BatchGenerator/RChunkLoader.hxx"
#include "TMVA/BatchGenerator/RBatchLoader.hxx"
#include "TMVA/BatchGenerator/RDatasetLoader.hxx"
#include "TMVA/BatchGenerator/RSampler.hxx"
#include "TMVA/BatchGenerator/RTensorOperations.hxx"
#include "TROOT.h"

#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <variant>
#include <vector>
#include <chrono>

namespace TMVA {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::TMVA::Experimental::Internal::RBatchGenerator
\ingroup tmva
\brief 

In this class, the processes of loading chunks (see RChunkLoader) and creating batches from those chunks (see RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset in an RDataFrame.
*/

template <typename... Args>
class RBatchGenerator {
private:
   std::vector<std::string> fCols;
   // clang-format on
   std::size_t fChunkSize;
   std::size_t fMaxChunks;
   std::size_t fBatchSize;
   std::size_t fBlockSize;
   std::size_t fNumColumns;
   std::size_t fNumDatasetCols;
   std::size_t fNumEntries;
   std::size_t fDNumEntries;
   std::size_t fSetSeed;
   std::size_t fSumVecSizes;

   std::size_t fNumDataFrames;
   std::vector<std::size_t> fNumEntriesDatasets;
   std::vector<std::size_t> fNumTrainingEntriesDatasets;
   std::vector<std::size_t> fNumValidationEntriesDatasets;
   
   std::size_t fSumNumEntries; 
   float fValidationSplit;

   std::string fSampleType;
   float fSampleStrategy;
   std::size_t fRandomState;
   std::unique_ptr<RChunkLoader<Args...>> fChunkLoader;
   std::unique_ptr<RDatasetLoader<Args...>> fDatasetLoader;
   std::unique_ptr<RBatchLoader> fBatchLoader;
   
   std::unique_ptr<RBatchLoader> fChunkTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fChunkValidationBatchLoader;
   
   std::unique_ptr<RBatchLoader> fTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fValidationBatchLoader;

   std::unique_ptr<RBatchLoader> fSampleTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fSampleValidationBatchLoader;
   std::unique_ptr<RTensorOperations> fTensorOperations;
   
   std::unique_ptr<RSampler> fSampler;
   std::unique_ptr<RSampler> fTrainingSampler;
   std::unique_ptr<RSampler> fValidationSampler;
   std::unique_ptr<std::thread> fLoadingThread;

   std::size_t fTrainingChunkNum;
   std::size_t fValidationChunkNum;

   std::vector<ROOT::RDF::RNode> f_rdfs;

   std::mutex fIsActiveMutex;

   bool fDropRemainder;
   bool fShuffle;
   bool fIsActive{false}; // Whether the loading thread is active
   bool fNotFiltered;
   bool fUseWholeFile;
   bool fLoadEager;
   bool fDataframeBalance{false};
   bool fQueue{true};
   bool fVector{false};
   bool fReplacement;
  
   bool fEpochActive{false};
   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   std::size_t fNumTrainingChunks;
   std::size_t fNumValidationChunks;

   TMVA::Experimental::RTensor<float> fTrainChunkTensor;
   TMVA::Experimental::RTensor<float> fValidationChunkTensor;

   std::vector<TMVA::Experimental::RTensor<float>> fTrainingDatasets;
   std::vector<TMVA::Experimental::RTensor<float>> fValidationDatasets;   

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;
  
public:
   RBatchGenerator(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t chunkSize = 0,
                   const std::size_t blockSize = 0, const std::size_t batchSize = 0, const std::vector<std::string> &cols = {},
                   const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0, const float validationSplit = 0.0, const std::size_t maxChunks = 0, bool shuffle = true,
                   bool dropRemainder = true, const std::size_t setSeed = 0, bool loadEager = false, std::string sampleType = "random", float sampleStrategy = 0.5, const std::size_t randomState = 0, bool replacement = false)

      : f_rdfs(rdfs),
        fCols(cols),
        fChunkSize(chunkSize),
        fBlockSize(blockSize),
        fBatchSize(batchSize),
        fValidationSplit(validationSplit),
        fSampleType(sampleType),
        fSampleStrategy(sampleStrategy),
        fRandomState(randomState),
        fReplacement(replacement),
        fMaxChunks(maxChunks),
        fDropRemainder(dropRemainder),
        fSetSeed(setSeed),
        fLoadEager(loadEager),
        fShuffle(shuffle),
        fNotFiltered(f_rdfs[0].GetFilterNames().empty()),
        fUseWholeFile(maxChunks == 0),
        fNumColumns(cols.size()),
        fTrainChunkTensor({0, 0}),
        fValidationChunkTensor({0, 0})
   {
      fDNumEntries = 1111;
      fTrainingSampler = std::make_unique<RSampler>(fSampleStrategy, fRandomState, fReplacement, fShuffle, fSetSeed, fLoadEager);
      fValidationSampler = std::make_unique<RSampler>(fSampleStrategy, fRandomState, fReplacement, fShuffle, fSetSeed, fLoadEager);
        
      fTensorOperations = std::make_unique<RTensorOperations>(fShuffle, fSetSeed);
      fBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, fDNumEntries, fDropRemainder);
      
      fNumDataFrames = f_rdfs.size();
      fSumVecSizes = std::accumulate(vecSizes.begin(), vecSizes.end(), 0);
      fNumDatasetCols = fNumColumns + fSumVecSizes - vecSizes.size();

      if (loadEager) {
         for (std::size_t i = 0; i < fNumDataFrames; i++) {
              std::size_t NumEntries = f_rdfs[i].Count().GetValue();
              ROOT::RDF::RResultPtr<std::vector<ULong64_t>> Entries = f_rdfs[i].Take<ULong64_t>("rdfentry_");
              // add the last element in entries to not go out of range when filling chunks
              Entries->push_back((*Entries)[NumEntries - 1] + 1);

              fDatasetLoader =
                 std::make_unique<RDatasetLoader<Args...>>(f_rdfs[i], fValidationSplit, fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
              
              TMVA::Experimental::RTensor<float> TrainingDataset({0, 0});
              TMVA::Experimental::RTensor<float> ValidationDataset({0, 0});   
              fDatasetLoader->SplitDataset(TrainingDataset, ValidationDataset);
              

              fTrainingDatasets.push_back(TrainingDataset);
              fValidationDatasets.push_back(ValidationDataset);
              fNumEntriesDatasets.push_back(NumEntries);                            
              fNumTrainingEntriesDatasets.push_back(TrainingDataset.GetShape()[0]);
              fNumValidationEntriesDatasets.push_back(ValidationDataset.GetShape()[0]);
         }

         if (sampleType == "random") {
           std::size_t NumTrainingEntries = std::accumulate(fNumTrainingEntriesDatasets.begin(), fNumTrainingEntriesDatasets.end(), 0);
           std::size_t NumValidationEntries = std::accumulate(fNumValidationEntriesDatasets.begin(), fNumValidationEntriesDatasets.end(), 0);
           
           fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
           fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);
         }

         else if (sampleType == "undersampling") {
           std::size_t NumTrainingEntries = fTrainingSampler->SetupRandomUnderSampler(fTrainingDatasets);
           std::size_t NumValidationEntries = fValidationSampler->SetupRandomUnderSampler(fValidationDatasets);

           fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
           fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);
         }

         else if (sampleType == "oversampling") {
           std::size_t NumTrainingEntries = fTrainingSampler->SetupRandomOverSampler(fTrainingDatasets);
           std::size_t NumValidationEntries = fValidationSampler->SetupRandomOverSampler(fValidationDatasets);

           fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
           fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);
         }
         
      }
      
      else {
         fChunkLoader =
            std::make_unique<RChunkLoader<Args...>>(f_rdfs[0], fChunkSize, fBlockSize, fValidationSplit,
                                                    fCols, vecSizes, vecPadding, fShuffle, fSetSeed);
          // split the dataset into training and validation sets
         fChunkLoader->SplitDataset();
         std::size_t NumTrainingEntries = fChunkLoader->GetNumTrainingEntries();
         std::size_t NumValidationEntries = fChunkLoader->GetNumValidationEntries();
          
         fChunkTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumTrainingEntries, fDropRemainder);
         fChunkValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fNumDatasetCols, NumValidationEntries, fDropRemainder);
         // number of training and validation chunks, calculated in RChunkConstructor
         fNumTrainingChunks = fChunkLoader->GetNumTrainingChunks();
         fNumValidationChunks = fChunkLoader->GetNumValidationChunks();

         fTrainingChunkNum = 0;
         fValidationChunkNum = 0;
      }
   }
      
   ~RBatchGenerator() { DeActivate(); }

   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = false;
      }

      fBatchLoader->DeActivate();

      if (fLoadingThread) {
         if (fLoadingThread->joinable()) {
            fLoadingThread->join();
         }
      }
   }

   /// \brief Activate the loading process by starting the batchloader, and
   /// spawning the loading thread.
   void Activate()
   {
      if (fIsActive)
         return;

      {
         std::lock_guard<std::mutex> lock(fIsActiveMutex);
         fIsActive = true;
      }
      fBatchLoader->Activate();
      // fLoadingThread = std::make_unique<std::thread>(&RBatchGenerator::LoadChunks, this);
   }

   void ActivateEpoch() { fEpochActive = true; }

   void DeActivateEpoch() { fEpochActive = false; }

   void ActivateTrainingEpoch() { fTrainingEpochActive = true; }

   void DeActivateTrainingEpoch() { fTrainingEpochActive = false; }

   void ActivateValidationEpoch() { fValidationEpochActive = true; }

   void DeActivateValidationEpoch() { fValidationEpochActive = false; }

   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see RBatchLoader)
   void CreateTrainBatches()
   {
     fTrainingEpochActive = true;
            
     if (fLoadEager) {
     
        TMVA::Experimental::RTensor<float> TrainingDataset({0, 0});
        if (fSampleType == "random") {
          fTrainingSampler->RandomSampler(TrainingDataset, fTrainingDatasets);
        }

        else if (fSampleType == "undersampling") {
          fTrainingSampler->RandomUnderSampler(TrainingDataset, fTrainingDatasets);
        }

        else if (fSampleType == "oversampling") {
          fTrainingSampler->RandomOverSampler(TrainingDataset, fTrainingDatasets);
        }
        
        fTrainingBatchLoader->CreateBatches(TrainingDataset, 1, fQueue);        
     }
     
     else {
       fTrainingChunkNum = 0;
       
       fChunkLoader->CreateTrainingChunksIntervals();
       fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
       
       std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
       fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fQueue);
       fTrainingChunkNum++;
     }
   }


   /// \brief Create training batches by first loading a chunk (see RChunkLoader) and split it into batches (see RBatchLoader)
   void CreateValidationBatches()
   {
      fValidationEpochActive = true;
      
      if (fLoadEager) {
         TMVA::Experimental::RTensor<float> ValidationDataset({0, 0});
         
         if (fSampleType == "random") {
             fValidationSampler->RandomSampler(ValidationDataset, fValidationDatasets);
           }
         
         else if (fSampleType == "undersampling") {
           fValidationSampler->RandomUnderSampler(ValidationDataset, fValidationDatasets);
         }

         else if (fSampleType == "oversampling") {
           fValidationSampler->RandomOverSampler(ValidationDataset, fValidationDatasets);
         }
         
         fValidationBatchLoader->CreateBatches(ValidationDataset, 1, fQueue);
      }

      else {
         fValidationChunkNum = 0;
         
         fChunkLoader->CreateValidationChunksIntervals();
         fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
         
         std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
         fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fQueue);
         fValidationChunkNum++;
      }
   }
   
   /// \brief Loads a training batch from the queue
   TMVA::Experimental::RTensor<float> GetTrainBatch()
   {
      if (fLoadEager) {
         // Get next batch if available
         return fTrainingBatchLoader->GetBatch();            
      }

      else {
         auto batchQueue = fChunkTrainingBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fTrainingChunkNum < fNumTrainingChunks) {
            fChunkLoader->LoadTrainingChunk(fTrainChunkTensor, fTrainingChunkNum);
            std::size_t lastTrainingBatch = fNumTrainingChunks - fTrainingChunkNum;
            fChunkTrainingBatchLoader->CreateBatches(fTrainChunkTensor, lastTrainingBatch, fQueue);            
            fTrainingChunkNum++;
         }

         else {
            std::size_t NumTrainingEntries = fChunkTrainingBatchLoader->GetNumEntries();
            std::size_t NumValidationEntries = fChunkValidationBatchLoader->GetNumEntries();
            std::size_t NumEntries = NumTrainingEntries + NumValidationEntries;
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdfs[0], 0, NumEntries);
         }

         // Get next batch if available
         return fChunkTrainingBatchLoader->GetBatch();
      }
   }

   /// \brief Loads a training batch from the queue
   TMVA::Experimental::RTensor<float> GetValidationBatch()
   {
      if (fLoadEager) {
         // Get next batch if available
         return fValidationBatchLoader->GetBatch();
      }

      else {
         auto batchQueue = fChunkValidationBatchLoader->GetNumBatchQueue();

         // load the next chunk if the queue is empty
         if (batchQueue < 1 && fValidationChunkNum < fNumValidationChunks) {
            fChunkLoader->LoadValidationChunk(fValidationChunkTensor, fValidationChunkNum);
            std::size_t lastValidationBatch = fNumValidationChunks - fValidationChunkNum;
            fChunkValidationBatchLoader->CreateBatches(fValidationChunkTensor, lastValidationBatch, fQueue);
            fValidationChunkNum++;
         }

         else {
            std::size_t NumTrainingEntries = fChunkTrainingBatchLoader->GetNumEntries();
            std::size_t NumValidationEntries = fChunkValidationBatchLoader->GetNumEntries();
            std::size_t NumEntries = NumTrainingEntries + NumValidationEntries;
            ROOT::Internal::RDF::ChangeBeginAndEndEntries(f_rdfs[0], 0, NumEntries);
         }

         // Get next batch if available
         return fChunkValidationBatchLoader->GetBatch();
      }
   }
  
   std::size_t NumberOfTrainingBatches() {
      if (fLoadEager) {
         return fTrainingBatchLoader->GetNumBatches();
      }         
      else {
         return fChunkTrainingBatchLoader->GetNumBatches();
      }         
   }

   std::size_t NumberOfValidationBatches() {
      if (fLoadEager) {
         return fValidationBatchLoader->GetNumBatches();
      }         
      else {
         return fChunkValidationBatchLoader->GetNumBatches();
      }         
   }
   
   std::size_t TrainRemainderRows() {
      if (fLoadEager) {
         return fTrainingBatchLoader->GetNumRemainderRows();
      }
      else {
         return fChunkTrainingBatchLoader->GetNumRemainderRows();
      }
   }

   std::size_t ValidationRemainderRows() {
      if (fLoadEager) {
         return fValidationBatchLoader->GetNumRemainderRows();
      }
      else {
         return fChunkValidationBatchLoader->GetNumRemainderRows();
      }
   }
   
   bool IsActive() { return fIsActive; }
   bool TrainingIsActive() { return fTrainingEpochActive; }
   /// \brief Returns the next batch of validation data if available.
   /// Returns empty RTensor otherwise.
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBATCHGENERATOR
