#ifndef GITRM_INPUT_HPP
#define GITRM_INPUT_HPP

#include <fstream>
#include <vector>

#include <libconfig.h++>

class GitrmInput {
public:
  GitrmInput(const std::string& f, bool debug=false);
  GitrmInput(const GitrmInput&) = delete;
  GitrmInput& operator=(const GitrmInput&) = delete;
  int parseConfigFile();
  template <typename T=std::string>
  T readConfigVar (const std::string& s);

  void testInputConfig();
private:
  const std::string& cfgFileName;
  libconfig::Config* cfg;
  int myRank = -1;
};


template <typename T>
T GitrmInput::readConfigVar(const std::string& s) {
  T var;
  if(cfg->lookupValue(s, var)) {
  } else {
    std::cout << "ERROR: Failed getting config " << s << std:: endl;
    exit(EXIT_FAILURE);
  }
  return var;
}

#endif

