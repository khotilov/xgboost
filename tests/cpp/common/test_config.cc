// Copyright by Contributors
#include <xgboost/logging.h>
#include "../../src/common/config.h"

#include "../helpers.h"

TEST(Common, ConfigStreamReader) {
  std::stringstream ss;
  std::vector<std::string> name, val;
  
  name.clear(); val.clear(); ss.flush(); ss.clear(); ss.str("");
  ss << "a=1\nb=2\tc=3\n d = 4   e=5 f='asd' g_1 = \" asd asd\"\n#this is comment\nblah= qwe";
  xgboost::common::ConfigStreamReader cr(ss);
  cr.Init();
  while (cr.Next()) {
    name.push_back(std::string(cr.name()));
    val.push_back(std::string(cr.val()));
  }
  //for (size_t i=0; i<name.size(); ++i) std::cout<<i<<":\t'"<<name[i]<<"':'"<<val[i]<<"'"<<std::endl;
  EXPECT_EQ(name[0], "a");
  EXPECT_EQ(val[0], "1");
  EXPECT_EQ(name[1], "b");
  EXPECT_EQ(val[1], "2");
  EXPECT_EQ(name[2], "c");
  EXPECT_EQ(val[2], "3");
  EXPECT_EQ(name[3], "d");
  EXPECT_EQ(val[3], "4");
  EXPECT_EQ(name[4], "e");
  EXPECT_EQ(val[4], "5");
  EXPECT_EQ(name[5], "f");
  EXPECT_EQ(val[5], "asd");
  EXPECT_EQ(name[6], "g_1");
  EXPECT_EQ(val[6], " asd asd");
  EXPECT_EQ(name[7], "blah");
  EXPECT_EQ(val[7], "qwe");
}
