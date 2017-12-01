#pragma once
#include "stdafx.h"
#include "nnet_loss.h"
#include "nnet_layer.h"


namespace nnet
{
  class nnet
  {
  public:
    field<layer> layers;
    field<mat> conns;
    field<mat> weights;
    field<mat> bufs;
    field<mat> dlosses;
    field<mat> deltas;
    field<mat> biases;
    field<mat> alphas;
    field<mat> lambdas;
    loss *los = NULL;
    int n_layers = 0;


    ~nnet() { if (los != NULL) delete[] los; }


    // conns: layer-connections of nnet, if n_layers = N, then n_conns = N + 1
    // layers: not include input-layer
    void init_malloc(const uvec &conns_dim)
    {
      n_layers = conns_dim.n_elem - 1;
      layers.set_size(n_layers);
      conns.set_size(conns_dim.n_elem);
      weights.set_size(n_layers);
      bufs.set_size(n_layers);
      dlosses.set_size(conns_dim.n_elem);
      deltas.set_size(n_layers);
      alphas.set_size(n_layers);
      lambdas.set_size(n_layers);
      biases.set_size(n_layers);

      int i = 0;
      weights.for_each([&i, &conns_dim](mat &m) { m.set_size(conns_dim(i + 1), conns_dim(i)); ++i; });
      i = 0;
      biases.for_each([&i, &conns_dim](mat &m) { m.set_size(conns_dim(i + 1), 1); ++i; });
      i = 0;
      alphas.for_each([&i, &conns_dim](mat &m) { m.set_size(conns_dim(i + 1), 1); ++i; });
      i = 0;
      lambdas.for_each([&i, &conns_dim](mat &m) { m.set_size(conns_dim(i + 1), 1); ++i; });
      i = 0;
      layers.for_each([&i, &conns_dim](layer &l) { l.init_malloc(conns_dim(i + i)); ++i; });
    }


    void train(const mat& x, const mat& y, const int &n_iterations, field<mat> *losses)
    {
      assert(los != NULL);

      conns(0) = x;
      if (losses != NULL)
        losses->set_size(n_iterations);

      for (int k = 0; k < n_iterations; ++k)
      {
        // propagate
        for (int l = 0; l < n_layers; ++l)
          layers(l).propagate(
            conns(l),
            weights(l),
            biases(l),
            &bufs(l),
            &conns(l + 1)
          );

        // k-th iteration's loss
        if (losses != NULL)
          losses->at(k) = los->eval(conns(n_layers), y);

        // dloss
        dlosses(n_layers) = los->diff(conns(n_layers), y);

        // back propagate
        for (int l = n_layers - 1; l >= 0; --l)
          layers(l).back_propagate(
            conns(l),
            bufs(l),
            dlosses(l + 1),
            alphas(l),
            lambdas(l),
            &dlosses(l),
            &deltas(l),
            &weights(l),
            &biases(l)
          );
      }
    }
  };
}