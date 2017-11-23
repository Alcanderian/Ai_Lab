#pragma once
#include "stdafx.h"


template<class T>
class channel
{
private:
  T *data = NULL;


public:
  channel() = default;
  channel(T &data) { this->data = &data; }
  channel(T *data) { this->data = data; }
  void bind(T &data) { this->data = &data; }
  void bind(T *data) { this->data = data; }
  void unbind() { data = NULL; }
  bool binded() { return data == NULL; }
  T &operator()() { return *data; }
};
