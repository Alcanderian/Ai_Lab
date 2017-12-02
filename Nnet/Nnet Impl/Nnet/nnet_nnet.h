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
    field<mat> ios;
    field<mat> weights;
    field<mat> muls;
    field<mat> dlosses;
    field<mat> deltas;
    field<mat> biases;
    field<mat> alphas;
    field<mat> lambdas;
    loss *los = NULL;
    int n_layers = 0;


    // ios: layer-io of nnet, if n_layers = N, then n_ios = N + 1
    // layers: not include input-layer
    void init_malloc(const uvec &ios_dim)
    {
      n_layers = ios_dim.n_elem - 1;
      layers.set_size(n_layers);
      ios.set_size(ios_dim.n_elem);
      weights.set_size(n_layers);
      muls.set_size(n_layers);
      dlosses.set_size(ios_dim.n_elem);
      deltas.set_size(n_layers);
      alphas.set_size(n_layers);
      lambdas.set_size(n_layers);
      biases.set_size(n_layers);

      int i = 0;
      weights.for_each([&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), ios_dim(i)); ++i; });
      i = 0;
      biases.for_each([&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), 1); ++i; });
      i = 0;
      alphas.for_each([&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), 1); ++i; });
      i = 0;
      lambdas.for_each([&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), 1); ++i; });
      i = 0;
      layers.for_each([&i, &ios_dim](layer &l) { l.init_malloc(ios_dim(i + i)); ++i; });
    }


    void train(const mat& x, const mat& y, const int &n_iterations, field<mat> *losses)
    {
      assert(los != NULL);

      int len = x.n_cols;

      // init ios(0), first input
      ios(0) = x;

      if (losses != NULL)
        losses->set_size(n_iterations);

      for (int k = 0; k < n_iterations; ++k)
      {
        // propagate
        for (int l = 0; l < n_layers; ++l)
          layers(l).propagate(
            ios(l),
            weights(l),
            biases(l),
            &muls(l),
            &ios(l + 1)
          );

        // k-th iteration's avg-loss
        if (losses != NULL)
          losses->at(k) = los->avg_eval(ios(n_layers), y);

        // dloss(l + 1)
        dlosses(n_layers) = los->diff(ios(n_layers), y);

        // back propagate
        for (int l = n_layers - 1; l >= 0; --l)
          layers(l).back_propagate(
            ios(l),
            muls(l),
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