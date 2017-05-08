/*!
 * Copyright 2015 by Contributors
 * \file dump_model.cc
 * \brief Implementation of helper utilities for DumpModel
 */
#include <xgboost/dump_model.h>
#include <iostream>
#include "./common/config.h"

namespace xgboost {

DMLC_REGISTER_PARAMETER(DumpModelParam);

void DumpModelParam::InitFromConfigString(std::string cfg_string) {
  std::stringstream ss;
  // for legacy code, allow for cfg_string to be simply a format label
  if (cfg_string == "json" || cfg_string == "text") {
    ss << "format=";
  }
  ss << cfg_string;
  common::ConfigStreamReader cr(ss);
  cr.Init();
  std::vector<std::pair<std::string, std::string> > cfg;
  while (cr.Next()) {
    cfg.push_back(std::make_pair(std::string(cr.name()), std::string(cr.val())));
  }
  this->InitAllowUnknown(cfg);
}
}  // namespace xgboost
