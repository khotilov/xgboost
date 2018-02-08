/*!
 * Copyright 2014 by Contributors
 * \file gblinear.cc
 * \brief Implementation of Linear booster, with L1/L2 regularization: Elastic Net
 *        the update rule is parallel coordinate descent (shotgun)
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <dmlc/timer.h>
#include <xgboost/gbm.h>
#include <xgboost/logging.h>
#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace xgboost {
namespace gbm {

DMLC_REGISTRY_FILE_TAG(gblinear);

// algorithm for weight updates
enum GBLinearMethod {
  kHogwild,
  kStagewise
};

// model parameter
struct GBLinearModelParam :public dmlc::Parameter<GBLinearModelParam> {
  // number of feature dimension
  unsigned num_feature;
  // number of output group
  int num_output_group;
  // reserved field
  int reserved[32];
  // constructor
  GBLinearModelParam() {
    std::memset(this, 0, sizeof(GBLinearModelParam));
  }
  DMLC_DECLARE_PARAMETER(GBLinearModelParam) {
    DMLC_DECLARE_FIELD(num_feature).set_lower_bound(0)
        .describe("Number of features used in classification.");
    DMLC_DECLARE_FIELD(num_output_group).set_lower_bound(1).set_default(1)
        .describe("Number of output groups in the setting.");
  }
};

// training parameter
struct GBLinearTrainParam : public dmlc::Parameter<GBLinearTrainParam> {
  /*! \brief learning_rate */
  float learning_rate;
  /*! \brief regularization weight for L2 norm */
  float reg_lambda;
  /*! \brief regularization weight for L1 norm */
  float reg_alpha;
  /*! \brief regularization weight for L2 norm in bias */
  float reg_lambda_bias;
  /*! \brief flag for whether the loss sum is scaled by 1/N */
  bool scaled_loss;
  /*! \brief choice of algorithm for weight updates */
  int gblinear_method;
  /*! \brief number of top features to update per boosting round in stagewise algorithm */
  unsigned top_n;
  // flag to print out detailed breakdown of runtime
  int debug_verbose;
  // declare parameters
  DMLC_DECLARE_PARAMETER(GBLinearTrainParam) {
    DMLC_DECLARE_FIELD(learning_rate).set_lower_bound(0.0f).set_default(1.0f)
        .describe("Learning rate of each update.");
    DMLC_DECLARE_FIELD(reg_lambda).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L2 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_alpha).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L1 regularization on weights.");
    DMLC_DECLARE_FIELD(reg_lambda_bias).set_lower_bound(0.0f).set_default(0.0f)
        .describe("L2 regularization on bias.");
    DMLC_DECLARE_FIELD(scaled_loss)
        .set_default(false)
        .describe("Flag for whether the loss sum term is scaled by 1/N, where N is # of samples. "
                  "Scaled loss is used, e.g., in sckit-learn and glmnet. Note that the equivalent "
                  "regularization parameters' magnitudes for non-scaled loss are N times larger.");
    DMLC_DECLARE_FIELD(gblinear_method)
        .set_default(kHogwild)
        .add_enum("hogwild", kHogwild)
        .add_enum("stagewise", kStagewise)
        .describe("Algorithm for weight updates during a boosting iteration:\n"
                  "  hogwild: lock-free cyclic gradient descent (CGD), updating all features\n"
                  "           in the order approximately guided by the order of columns;\n"
                  "  stagewise: stagewise CGD, selecting and updating top-N features\n."
                  "           in the order of decreasing magnitude of univariate weight changes.");
    DMLC_DECLARE_FIELD(top_n)
        .set_lower_bound(0u)
        .set_default(1u)
        .describe("The number of top features to select and update per boosting round "
                  "when using stagewise algorithm. Zero means using all the features ordered "
                  "by decreasing magnitude of their univariate weight changes.");
    DMLC_DECLARE_FIELD(debug_verbose)
        .set_lower_bound(0)
        .set_default(0)
        .describe("Flag to print out detailed breakdown of runtime.");
    // alias of parameters
    DMLC_DECLARE_ALIAS(learning_rate, eta);
    DMLC_DECLARE_ALIAS(reg_lambda, lambda);
    DMLC_DECLARE_ALIAS(reg_alpha, alpha);
    DMLC_DECLARE_ALIAS(reg_lambda_bias, lambda_bias);
  }
  // given original weight calculate delta
  inline double CalcDelta(double sum_grad, double sum_hess, double w) const {
    if (sum_hess < 1e-5f) return 0.0f;
    double tmp = w - (sum_grad + reg_lambda * w) / (sum_hess + reg_lambda);
    if (tmp >=0) {
      return std::max(-(sum_grad + reg_lambda * w + reg_alpha) / (sum_hess + reg_lambda), -w);
    } else {
      return std::min(-(sum_grad + reg_lambda * w - reg_alpha) / (sum_hess + reg_lambda), -w);
    }
  }
  // given original weight calculate delta bias
  inline double CalcDeltaBias(double sum_grad, double sum_hess, double w) const {
    return - (sum_grad + reg_lambda_bias * w) / (sum_hess + reg_lambda_bias);
  }
};

/*!
 * \brief gradient boosted linear model
 */
class GBLinear : public GradientBooster {
 public:
  explicit GBLinear(bst_float base_margin)
      : base_margin_(base_margin) {
  }

  void Configure(const std::vector<std::pair<std::string, std::string> >& cfg) override {
    if (model.weight.size() == 0) {
      model.param.InitAllowUnknown(cfg);
    }
    param.InitAllowUnknown(cfg);
  }

  void Load(dmlc::Stream* fi) override {
    model.Load(fi);
  }

  void Save(dmlc::Stream* fo) const override {
    model.Save(fo);
  }

  void DoBoost(DMatrix *p_fmat,
               std::vector<bst_gpair> *in_gpair,
               ObjFunction* obj) override {
    // lazily initialize the model when not ready.
    if (model.weight.size() == 0) {
      model.InitModel();

      if (param.gblinear_method == kStagewise) {
        // pre-allocate the memory
        deltaw.reserve(model.param.num_feature * model.param.num_output_group);
        sorted_idx.reserve(model.param.num_feature);
        top_features.reserve(model.param.num_feature);
      }
    }
    double tstart = dmlc::GetTime();
    switch (param.gblinear_method) {
      case kHogwild: {
        UpdateHogwild(p_fmat, in_gpair);
        break;
      }
      case kStagewise: {
        UpdateStagewise(p_fmat, in_gpair);
        break;
      }
    }
    if (param.debug_verbose > 0) {
      LOG(CONSOLE) << "DoBoost(): " << 1000 * (dmlc::GetTime() - tstart) << " msec";
    }
  }

  void PredictBatch(DMatrix *p_fmat,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit) override {
    if (model.weight.size() == 0) {
      model.InitModel();
    }
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::Predict ntree_limit!=0 is not valid for gblinear";
    std::vector<bst_float> &preds = *out_preds;
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    preds.resize(0);
    // start collecting the prediction
    dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
    const int ngroup = model.param.num_output_group;
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      CHECK_EQ(batch.base_rowid * ngroup, preds.size());
      // output convention: nrow * k, where nrow is number of rows
      // k is number of group
      preds.resize(preds.size() + batch.size * ngroup);
      // parallel over local batch
      const omp_ulong nsize = static_cast<omp_ulong>(batch.size);
      #pragma omp parallel for schedule(static)
      for (omp_ulong i = 0; i < nsize; ++i) {
        const size_t ridx = batch.base_rowid + i;
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float margin =  (base_margin.size() != 0) ?
              base_margin[ridx * ngroup + gid] : base_margin_;
          this->Pred(batch[i], &preds[ridx * ngroup], gid, margin);
        }
      }
    }
  }

  // add base margin
  void PredictInstance(const SparseBatch::Inst &inst,
               std::vector<bst_float> *out_preds,
               unsigned ntree_limit,
               unsigned root_index) override {
    const int ngroup = model.param.num_output_group;
    for (int gid = 0; gid < ngroup; ++gid) {
      this->Pred(inst, dmlc::BeginPtr(*out_preds), gid, base_margin_);
    }
  }

  void PredictLeaf(DMatrix *p_fmat,
                   std::vector<bst_float> *out_preds,
                   unsigned ntree_limit) override {
    LOG(FATAL) << "gblinear does not support prediction of leaf index";
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate, int condition = 0,
                           unsigned condition_feature = 0) override {
    if (model.weight.size() == 0) {
      model.InitModel();
    }
    CHECK_EQ(ntree_limit, 0U)
        << "GBLinear::PredictContribution: ntrees is only valid for gbtree predictor";
    const std::vector<bst_float>& base_margin = p_fmat->info().base_margin;
    const int ngroup = model.param.num_output_group;
    const size_t ncolumns = model.param.num_feature + 1;
    // allocate space for (#features + bias) times #groups times #rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(p_fmat->info().num_row * ncolumns * ngroup);
    // make sure contributions is zeroed, we could be reusing a previously allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
    // start collecting the contributions
    dmlc::DataIter<RowBatch>* iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      // parallel over local batch
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const RowBatch::Inst &inst = batch[i];
        size_t row_idx = static_cast<size_t>(batch.base_rowid + i);
        // loop over output groups
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float *p_contribs = &contribs[(row_idx * ngroup + gid) * ncolumns];
          // calculate linear terms' contributions
          for (bst_uint c = 0; c < inst.length; ++c) {
            if (inst[c].index >= model.param.num_feature) continue;
            p_contribs[inst[c].index] = inst[c].fvalue * model[inst[c].index][gid];
          }
          // add base margin to BIAS
          p_contribs[ncolumns - 1] = model.bias()[gid] +
            ((base_margin.size() != 0) ? base_margin[row_idx * ngroup + gid] : base_margin_);
        }
      }
    }
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           unsigned ntree_limit, bool approximate) override {
    LOG(FATAL) << "gblinear does not provide interaction contributions";
  }

  std::vector<std::string> DumpModel(const FeatureMap& fmap,
                                     bool with_stats,
                                     std::string format) const override {
    const int ngroup = model.param.num_output_group;
    const unsigned nfeature = model.param.num_feature;

    std::stringstream fo("");
    if (format == "json") {
      fo << "  { \"bias\": [" << std::endl;
      for (int gid = 0; gid < ngroup; ++gid) {
        if (gid != 0) fo << "," << std::endl;
        fo << "      " << model.bias()[gid];
      }
      fo << std::endl << "    ]," << std::endl
         << "    \"weight\": [" << std::endl;
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          if (i != 0 || gid != 0) fo << "," << std::endl;
          fo << "      " << model[i][gid];
        }
      }
      fo << std::endl << "    ]" << std::endl << "  }";
    } else {
      fo << "bias:\n";
      for (int gid = 0; gid < ngroup; ++gid) {
        fo << model.bias()[gid] << std::endl;
      }
      fo << "weight:\n";
      for (unsigned i = 0; i < nfeature; ++i) {
        for (int gid = 0; gid < ngroup; ++gid) {
          fo << model[i][gid] << std::endl;
        }
      }
    }
    std::vector<std::string> v;
    v.push_back(fo.str());
    return v;
  }

 protected:

  void UpdateBias(DMatrix *p_fmat,
                  std::vector<bst_gpair> *in_gpair) {
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    const RowSet &rowset = p_fmat->buffered_rowset();
    // for all the output group
    for (int gid = 0; gid < ngroup; ++gid) {
      double sum_grad = 0.0, sum_hess = 0.0;
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static) reduction(+: sum_grad, sum_hess)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          sum_grad += p.grad; sum_hess += p.hess;
        }
      }
      if (param.scaled_loss && ndata > 0) {
        sum_grad /= ndata; sum_hess /= ndata;
      }
      // remove bias effect
      bst_float dw = static_cast<bst_float>(
        param.learning_rate * param.CalcDeltaBias(sum_grad, sum_hess, model.bias()[gid]));
      model.bias()[gid] += dw;
      // update grad value
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        bst_gpair &p = gpair[rowset[i] * ngroup + gid];
        if (p.hess >= 0.0f) {
          p.grad += p.hess * dw;
        }
      }
    }
  }

  void UpdateHogwild(DMatrix *p_fmat,
               std::vector<bst_gpair> *in_gpair) {
    this->UpdateBias(p_fmat, in_gpair);
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    while (iter->Next()) {
      const ColBatch &batch = iter->Value();
      // number of features
      const bst_omp_uint nfeat = static_cast<bst_omp_uint>(batch.size);
      // lock-free parallel updates of all features
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const bst_uint fid = batch.col_index[i];
        ColBatch::Inst col = batch[i];
        for (int gid = 0; gid < ngroup; ++gid) {
          const bst_uint ndata = col.length;
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < ndata; ++j) {
            const bst_float v = col[j].fvalue;
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            sum_grad += p.grad * v;
            sum_hess += p.hess * v * v;
          }
          if (param.scaled_loss && ndata > 0) {
            sum_grad /= ndata; sum_hess /= ndata;
          }
          bst_float &w = model[fid][gid];
          bst_float dw = static_cast<bst_float>(param.learning_rate *
                                                param.CalcDelta(sum_grad, sum_hess, w));
          w += dw;
          // update grad values
          for (bst_uint j = 0; j < col.length; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            p.grad += p.hess * col[j].fvalue * dw;
          }
        }
      }
    }
  }

  void UpdateStagewise(DMatrix *p_fmat,
                           std::vector<bst_gpair> *in_gpair) {
    this->UpdateBias(p_fmat, in_gpair);
    std::vector<bst_gpair> &gpair = *in_gpair;
    const int ngroup = model.param.num_output_group;
    const unsigned nfeature = model.param.num_feature;

    // Calculate univariate weight changes
    deltaw.resize(nfeature * ngroup);
    std::fill(deltaw.begin(), deltaw.end(), 0.f);
    dmlc::DataIter<ColBatch> *iter = p_fmat->ColIterator();
    while (iter->Next()) {
      const ColBatch &batch = iter->Value();
      const bst_omp_uint nfeat = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nfeat; ++i) {
        const bst_uint fid = batch.col_index[i];
        ColBatch::Inst col = batch[i];
        for (int gid = 0; gid < ngroup; ++gid) {
          const bst_uint ndata = col.length;
          double sum_grad = 0.0, sum_hess = 0.0;
          for (bst_uint j = 0; j < ndata; ++j) {
            const bst_float v = col[j].fvalue;
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            sum_grad += p.grad * v;
            sum_hess += p.hess * v * v;
          }
          if (param.scaled_loss && ndata > 0) {
            sum_grad /= ndata; sum_hess /= ndata;
          }
          bst_float dw = static_cast<bst_float>(param.learning_rate *
                           param.CalcDelta(sum_grad, sum_hess, model[fid][gid]));
          if (dw != 0) deltaw[gid * nfeature + fid] = dw;
        }
      }
    }

    // Update top-N features that have the largest univariate abs weight changes
    const unsigned max_n = param.top_n > 0 ? param.top_n : nfeature;
    bst_float *pdeltaw = &deltaw[0];
    for (int gid = 0; gid < ngroup; ++gid) {
      const size_t g_offset = gid * nfeature;
      // get the indices of sorted deltaw (in descending order of abs values)
      sorted_idx.resize(nfeature);
      std::iota(sorted_idx.begin(), sorted_idx.end(), g_offset);
      std::sort(sorted_idx.begin(), sorted_idx.end(),
                [pdeltaw](size_t i, size_t j) {
                  return std::abs(*(pdeltaw + i)) > std::abs(*(pdeltaw + j));
                });
      bst_float max_dw = deltaw[sorted_idx[0]];
      if (max_dw == 0.f) continue;
      // Pick top-N features, and do cyclic updates only for them.
      // NOTE: a possibility of several features having the same weight is ignored.
      top_features.clear();
      for (auto &i: sorted_idx) {
        if (deltaw[i] == 0.f || top_features.size() == max_n) break;
        top_features.push_back(i - g_offset);
      }
      iter = p_fmat->ColIterator(top_features);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        const bst_uint nfeat = static_cast<bst_omp_uint>(batch.size);
        for (bst_uint i = 0; i < nfeat; ++i) {
          const bst_uint fid = batch.col_index[i];
          ColBatch::Inst col = batch[i];
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          bst_float &w = model[fid][gid];
          bst_float dw;
          if (i == 0) {
            dw = max_dw; // optimization for the 1st feature: no need to update dw
          } else {
            double sum_grad = 0.0, sum_hess = 0.0;
            #pragma omp parallel for schedule(static) reduction(+: sum_grad, sum_hess)
            for (bst_omp_uint j = 0; j < ndata; ++j) {
              const bst_float v = col[j].fvalue;
              bst_gpair &p = gpair[col[j].index * ngroup + gid];
              if (p.hess < 0.0f) continue;
              sum_grad += p.grad * v;
              sum_hess += p.hess * v * v;
            }
            if (param.scaled_loss && ndata > 0) {
              sum_grad /= ndata; sum_hess /= ndata;
            }
            dw = static_cast<bst_float>(param.learning_rate *
                                        param.CalcDelta(sum_grad, sum_hess, w));
          }
          w += dw;
          // update grad values
          #pragma omp parallel for schedule(static)
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            bst_gpair &p = gpair[col[j].index * ngroup + gid];
            if (p.hess < 0.0f) continue;
            p.grad += p.hess * col[j].fvalue * dw;
          }
        }
      }
    }
  }

  inline void Pred(const RowBatch::Inst &inst, bst_float *preds, int gid, bst_float base) {
    bst_float psum = model.bias()[gid] + base;
    for (bst_uint i = 0; i < inst.length; ++i) {
      if (inst[i].index >= model.param.num_feature) continue;
      psum += inst[i].fvalue * model[inst[i].index][gid];
    }
    preds[gid] = psum;
  }

  // model for linear booster
  class Model {
   public:
    // parameter
    GBLinearModelParam param;
    // weight for each of feature, bias is the last one
    std::vector<bst_float> weight;
    // initialize the model parameter
    inline void InitModel(void) {
      // bias is the last weight
      weight.resize((param.num_feature + 1) * param.num_output_group);
      std::fill(weight.begin(), weight.end(), 0.0f);
    }
    // save the model to file
    inline void Save(dmlc::Stream* fo) const {
      fo->Write(&param, sizeof(param));
      fo->Write(weight);
    }
    // load model from file
    inline void Load(dmlc::Stream* fi) {
      CHECK_EQ(fi->Read(&param, sizeof(param)), sizeof(param));
      fi->Read(&weight);
    }
    // model bias
    inline bst_float* bias() {
      return &weight[param.num_feature * param.num_output_group];
    }
    inline const bst_float* bias() const {
      return &weight[param.num_feature * param.num_output_group];
    }
    // get i-th weight
    inline bst_float* operator[](size_t i) {
      return &weight[i * param.num_output_group];
    }
    inline const bst_float* operator[](size_t i) const {
      return &weight[i * param.num_output_group];
    }
  };
  // biase margin score
  bst_float base_margin_;
  // model field
  Model model;
  // training parameter
  GBLinearTrainParam param;
  // Per feature: shuffle index of each feature index
  std::vector<bst_uint> feat_index;

  // The following three are used by the stagewise algorithm.
  // Having them as class members helps to avoid repeated construction & memory allocs.
  std::vector<bst_float> deltaw;
  std::vector<size_t> sorted_idx;
  std::vector<bst_uint> top_features;
};

// register the objective functions
DMLC_REGISTER_PARAMETER(GBLinearModelParam);
DMLC_REGISTER_PARAMETER(GBLinearTrainParam);

XGBOOST_REGISTER_GBM(GBLinear, "gblinear")
.describe("Linear booster, implement generalized linear model.")
.set_body([](const std::vector<std::shared_ptr<DMatrix> >&cache, bst_float base_margin) {
    return new GBLinear(base_margin);
  });
}  // namespace gbm
}  // namespace xgboost
