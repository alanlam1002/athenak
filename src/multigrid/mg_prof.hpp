#ifndef MULTIGRID_MG_PROF_HPP_
#define MULTIGRID_MG_PROF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mg_prof.hpp
//! \brief CFC/multigrid stage profiler (review V-001 A0). Off unless <cfc> mg_log >= 3.
//!
//! Fenced wall-clock timers, EXCLUSIVE accounting: a timer started while another is
//! running charges its time to itself and removes it from the enclosing timer, so the
//! exclusive times of all stages add up to the outermost enclosing time (no double
//! counting). Rows = solver (X, psi, alpha_psi, beta, other CFC), columns = stage.
//! When off (the default), MGProfTimer's constructor/destructor are a single bool test:
//! no fence, no clock read, no change to any kernel or to the order of any operation.
//! Report(): one block per solve set, MPI max/mean over ranks, printed by rank 0.

#include <chrono>
#include <string>
#include <vector>

namespace mgprof {

enum Solver : int { kX = 0, kPsi, kAlphaPsi, kBeta, kCFC, kNSolver };

enum Stage : int {
  // envelopes (their exclusive time is loop/driver overhead between the children)
  sSolveCall = 0,   // pmgd_->Solve(): PrepareForAMR + coeff + setup + SolveMG
  sIterative,       // SolveIterative() body outside V-cycles and norms
  sVCycle,          // SolveVCycle() body outside the level steps
  sTaskList,        // pdriver->ExecuteTaskList (exclusive = task-list scheduling)
  // set-up per solve
  sPrepareAMR,      // PrepareForAMR
  sCoeff,           // RestrictCoefficients / TransferCoeffToRoot / RestrictCoeffOctets
  sSetup,           // SetupMultigrid (+ multipole coefficients for X/beta)
  // MeshBlock-level MG tasks
  sSmooth,          // SmoothRed/SmoothBlack tasks (MeshBlock levels)
  sFASRHS,          // CalculateFASRHS task (StoreOldData + FAS RHS)
  sRestrict,        // Restrict task
  sProlong,         // Prolongate / FMGProlongate tasks
  sBndSend,         // SendBoundary task (pack + Isend + post Irecv)
  sBndRecv,         // RecvBoundary task (test/wait + unpack), every poll
  sBndOther,        // (unused since the split below)
  sBndStart,        // StartReceive task
  sBndFill,         // FillCoarseBoundary / FillFCBoundary tasks
  sBndProlFC,       // ProlongateFCBoundary task (fine-coarse ghost prolongation)
  sBndPhys,         // PhysicalBoundary task (outer BC: multipole / robin / zero)
  sBndClear,        // ClearSend / ClearRecv tasks (wait for sends / receives)
  // convergence norms
  sNormDev,         // device defect (CalculateDefectPack) + reduction, per channel
  sNormMPI,         // MPI_Allreduce of the norm, per channel
  // root transfer
  sToRootDev,       // SaveToRoot kernel + D->H sync of rootbuf
  sToRootMPI,       // MPI_Allgatherv loop (2*nvar or nvar calls)
  sToRootHost,      // host scatter into root/octets + H->D sync
  sFromRootOct,     // RestrictOctetsBeforeTransfer + SetOctetBoundariesBeforeTransfer
  sFromRootSet,     // Multigrid::SetFromRootGrid (root/octets -> block coarsest)
  sRootSync,        // SyncRootToHost / SyncRootToDevice copies
  // coarse levels (host or device)
  sOctet,           // octet-level smoothing/restrict/prolong/boundaries (host loops)
  sRoot,            // root-grid levels incl. SolveCoarsestGrid, MGRootBoundary
  sLog,             // mg_log LogSolve
  // CFC work outside the four MG solves
  cSource,          // AssembleVectorSource / u_plus_2s / matter source + coeff loads
  cLoad,            // LoadPoissonSource / LoadMatterSource / Load*Coefficient / seed
  cRetrieve,        // RetrieveSolution
  cMetric,          // AssembleConformalMetric (inside SolveConformalFactor)
  cADual,           // ComputeADual
  cRescale,         // RescaleMatterSources
  cRecon,           // ReconstructShift
  cAssemble,        // AssembleADM (AssembleLapseShiftK)
  cExchange,        // Rest/Send/Recv/Prolong/BCS ghost-exchange tasks (+ AssemblePsiG)
  cInitClear,       // CFC InitRecv / ClearSend / ClearRecv tasks
  kNStage
};

extern bool on;   // set once from <cfc> mg_log >= 3

void Start(int solver, int stage);
void Stop();
void Count(int solver, int stage, long n);    // extra counter (no time)
void CycleWall(int ncycle, double sec, bool solve_cycle);
void Report(int ncycle);                      // collective; prints and resets
const char *StageName(int s);

class Timer {
 public:
  Timer(int solver, int stage) {
    if (on) Start(solver, stage);
  }
  ~Timer() {
    if (on) Stop();
  }
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
};

int SolverId(const std::string &name);

}  // namespace mgprof

#endif  // MULTIGRID_MG_PROF_HPP_
