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
    field<loss*> loss_itf;
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
      loss_itf.set_size(ios_dim(ios_dim.n_elem - 1));

      loss_itf.fill(NULL);
      int i = 0;
      weights.for_each(
        [&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), ios_dim(i)); ++i; }
      );
      auto mat_malloc = [&i, &ios_dim](mat &m) { m.set_size(ios_dim(i + 1), 1); ++i; };
      i = 0;
      biases.for_each(mat_malloc);
      i = 0;
      alphas.for_each(mat_malloc);
      i = 0;
      lambdas.for_each(mat_malloc);
      i = 0;
      layers.for_each(
        [&i, &ios_dim](layer &l) { l.init_malloc(ios_dim(i + i)); ++i; }
      );
    }


    void train(const mat& x, const mat& y, const int &n_iterations, field<mat> *losses)
    {
      loss_itf.for_each(
        [](loss* &f) { assert(f != NULL); }
      );

      int len = y.n_cols, out_dim = y.n_rows;

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
        int i = 0;
        if (losses != NULL) {
          losses->at(k).set_size(out_dim, 1);
          losses->at(k).each_row(
            [&i, this, &y](mat &r) { r = this->loss_itf(i)->avg_eval(this->ios(this->n_layers).row(i), y.row(i)); ++i; }
          );
        }

        // dloss(l + 1)
        i = 0;
        if (dlosses(n_layers).n_rows != y.n_rows || dlosses(n_layers).n_cols != y.n_cols)
          dlosses(n_layers).set_size(y.n_rows, y.n_cols);
        dlosses(n_layers).each_row(
          [&i, this, &y](mat &r) { r = this->loss_itf(i)->diff(this->ios(this->n_layers).row(i), y.row(i)); ++i; }
        );

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
