/*!
 * Copyright 2017 by Contributors
 * \file dump_model.h
 * \brief helper utilities for DumpModel
 */
#ifndef XGBOOST_DUMP_MODEL_H_
#define XGBOOST_DUMP_MODEL_H_

#include <dmlc/parameter.h>
#include <iomanip>

namespace xgboost {

enum DumpModelFormat {
  kTEXT,
  kJSON
};

struct DumpModelParam : public dmlc::Parameter<DumpModelParam> {
  /*! \brief dump format */
  int format;
  /*! \brief numeric precision */
  int precision;
  /*! \brief output extra statistics */
  bool with_stats;
  // declare parameters
  DMLC_DECLARE_PARAMETER(DumpModelParam) {
    DMLC_DECLARE_FIELD(format)
        .set_default(kTEXT)
        .add_enum("text", kTEXT)
        .add_enum("json", kJSON)
        .describe("Dump format.");
    DMLC_DECLARE_FIELD(precision)
        .set_default(0)
        .set_lower_bound(0)
        .describe("Decimal precision of floating point output."\
                  " When set to 0, system default iostream precision is used.");
    DMLC_DECLARE_FIELD(with_stats)
        .set_default(false)
        .describe("Whether to dump out extra statistics as well (tree models only).");
  }
  /*!
   * \brief initialize from a string
   * \param cfg_string a string containg key-value pairs for parameters
   */
  void InitFromConfigString(std::string cfg_string);
  
  /*!
   * \brief set specific decimal precision if precision is not zero
   * \param ss dump''s stringstream
   */
  inline void setprecision(std::stringstream& ss) {
    if (precision > 0) {
      ss << std::setprecision(precision);
    }
  }
};

}  // namespace xgboost
#endif  // XGBOOST_DUMP_MODEL_H_
