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

#ifndef TMVA_RBATCHLOADER
#define TMVA_RBATCHLOADER

#include <vector>
#include <memory>
#include <numeric>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"

namespace TMVA::Experimental::Internal {

/**
\class ROOT::TMVA::Experimental::Internal::RBatchLoader
\ingroup tmva
\brief Building and loading the batches from loaded chunks in RChunkLoader

In this class the chunks that are loaded into memory (see RChunkLoader) are split into batches used in the ML training
which are loaded into a queue. This is done for both the training and validation chunks separately.
*/

class RBatchLoader {
private:
   std::size_t fNumEntries;
   std::size_t fNumColumns;   
   
   std::size_t fBatchSize;
   std::size_t fLeftoverBatchSize;
   std::size_t fNumFullBatches;
   std::size_t fNumLeftoverBatches;
   std::size_t fNumBatches;



   bool fIsActive = false;
   bool fDropRemainder;
  
   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   // queuse of tensors of the batches
   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fBatchQueue;

   // number of batches in the queue
   std::size_t fNumBatchQueue;

   // current batch that is loaded into memory
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   // primary and secondary batches used to create batches from a chunk
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fPrimaryLeftoverBatch;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fSecondaryLeftoverBatch;

   std::vector<TMVA::Experimental::RTensor<float>> fBatchesVector;
public:
   RBatchLoader(std::size_t batchSize, std::size_t numColumns, std::size_t numEntries, bool dropRemainder)
    : fBatchSize(batchSize),
      fNumColumns(numColumns),
      fNumEntries(numEntries),
      fDropRemainder(dropRemainder)
   {

     if (fBatchSize == 0) {
       fBatchSize = fNumEntries;
     }
     
     fLeftoverBatchSize = fNumEntries % fBatchSize;
     fNumFullBatches = fNumEntries / fBatchSize;

     fNumLeftoverBatches = fLeftoverBatchSize == 0 ? 0 : 1;

     if (fDropRemainder) {
       fNumBatches = fNumFullBatches;
     }
     
     else {
       fNumBatches = fNumFullBatches + fNumLeftoverBatches;
     }

     fPrimaryLeftoverBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      fSecondaryLeftoverBatch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});

      fNumBatchQueue = fBatchQueue.size();
   }

public:
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = true;
      }
      fBatchCondition.notify_all();
   }

   /// \brief DeActivate the batchloader. This means that no more batches are created.
   /// Batches can still be returned if they are already loaded
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = false;
      }
      fBatchCondition.notify_all();
   }

   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] idxs Index of batch in the chunk
   /// \return Training batch
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t idxs)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));
      std::copy(chunkTensor.GetData() + (idxs * fBatchSize * fNumColumns),
                chunkTensor.GetData() + ((idxs + 1) * fBatchSize * fNumColumns), batch->GetData());

      return batch;
   }

   /// \brief Loading the validation batch from the queue
   /// \return Training batch
   TMVA::Experimental::RTensor<float> GetBatch()
   {

      if (fBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fBatchQueue.front());
      fBatchQueue.pop();

      return *fCurrentBatch;
   }

   /// \brief Creating the validation batches from a chunk and adding them to the queue
   /// \param[in] chunkTensor RTensor with the data from the chunk
   /// \param[in] lastbatch Check if the batch in the chunk is the last one
   /// \param[in] leftoverBatchSize Size of the leftover batch in the validation dataset
   /// \param[in] dromRemainder Bool to drop the remainder batch or not
   void CreateBatches(TMVA::Experimental::RTensor<float> &chunkTensor, std::size_t lastbatch, bool Queue)
   {
      std::size_t ChunkSize = chunkTensor.GetShape()[0];
      std::size_t NumCols = chunkTensor.GetShape()[1];
      std::size_t Batches = ChunkSize / fBatchSize;
      std::size_t LeftoverBatchSize = ChunkSize % fBatchSize;

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      for (std::size_t i = 0; i < Batches; i++) {
         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, i));
      }

      TMVA::Experimental::RTensor<float> LeftoverBatch({LeftoverBatchSize, NumCols});
      std::copy(chunkTensor.GetData() + (Batches * fBatchSize * NumCols),
                chunkTensor.GetData() + (Batches * fBatchSize * NumCols + LeftoverBatchSize * NumCols),
                LeftoverBatch.GetData());

      std::size_t PrimaryLeftoverSize = (*fPrimaryLeftoverBatch).GetShape()[0];
      std::size_t emptySlots = fBatchSize - PrimaryLeftoverSize;

      if (emptySlots >= LeftoverBatchSize) {
         (*fPrimaryLeftoverBatch) =
            (*fPrimaryLeftoverBatch).Resize({PrimaryLeftoverSize + LeftoverBatchSize, NumCols});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (LeftoverBatchSize * NumCols),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * NumCols));

         if (emptySlots == LeftoverBatchSize) {
            auto copy =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
            std::copy(fPrimaryLeftoverBatch->GetData(),
                      fPrimaryLeftoverBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
            batches.emplace_back(std::move(copy));
            *fPrimaryLeftoverBatch = *fSecondaryLeftoverBatch;
            fSecondaryLeftoverBatch =
               std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         }
      }

      else if (emptySlots < LeftoverBatchSize) {
         (*fPrimaryLeftoverBatch) = (*fPrimaryLeftoverBatch).Resize({fBatchSize, NumCols});
         std::copy(LeftoverBatch.GetData(), LeftoverBatch.GetData() + (emptySlots * NumCols),
                   fPrimaryLeftoverBatch->GetData() + (PrimaryLeftoverSize * NumCols));
         (*fSecondaryLeftoverBatch) =
            (*fSecondaryLeftoverBatch).Resize({LeftoverBatchSize - emptySlots, NumCols});
         std::copy(LeftoverBatch.GetData() + (emptySlots * NumCols),
                   LeftoverBatch.GetData() + (LeftoverBatchSize * NumCols),
                   fSecondaryLeftoverBatch->GetData());
         auto copy =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{fBatchSize, fNumColumns});
         std::copy(fPrimaryLeftoverBatch->GetData(),
                   fPrimaryLeftoverBatch->GetData() + (fBatchSize * fNumColumns), copy->GetData());
         batches.emplace_back(std::move(copy));
         *fPrimaryLeftoverBatch = *fSecondaryLeftoverBatch;
         fSecondaryLeftoverBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      if (lastbatch == 1) {

         if (fDropRemainder == false && fLeftoverBatchSize > 0) {
            auto copy = std::make_unique<TMVA::Experimental::RTensor<float>>(
               std::vector<std::size_t>{fLeftoverBatchSize, fNumColumns});
            std::copy((*fPrimaryLeftoverBatch).GetData(),
                      (*fPrimaryLeftoverBatch).GetData() + (fLeftoverBatchSize * fNumColumns),
                      copy->GetData());
            batches.emplace_back(std::move(copy));
         }
         fPrimaryLeftoverBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
         fSecondaryLeftoverBatch =
            std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>{0, fNumColumns});
      }

      // batches = batches;
      for (std::size_t i = 0; i < batches.size(); i++) {
        if (Queue) {
          fBatchQueue.push(std::move(batches[i]));          
        }
        else {
          fBatchesVector.push_back(std::move(*batches[i]));
        }

      }
      
   }

  // void FillBatchesInQueue() {
  //   for (std::size_t i = 0; i < batches.size(); i++) {
  //     fBatchQueue.push(std::move(batches[i]));
  //   }
  // }

  std::size_t GetNumBatchQueue() { return fBatchQueue.size(); }
  std::size_t GetNumRemainderRows() { return fLeftoverBatchSize; }
  std::size_t GetNumBatches() { return fNumBatches; }
  std::size_t GetNumEntries() { return fNumEntries; }   
  
  std::vector<TMVA::Experimental::RTensor<float>> GetBatchesVector() {
    return fBatchesVector;
  }  
  // std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> GetBatchesVector() {return batches; }
  
};

} // namespace TMVA::Experimental::Internal

#endif // TMVA_RBATCHLOADER
